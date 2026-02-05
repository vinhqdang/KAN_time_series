import asyncio
import time
import json
import torch
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
import numpy as np

from .config import KAFKA_BOOTSTRAP, TOPIC, ASSETS, WINDOW, MAX_SCORED, DEVICE

async def ensure_topic(name, bootstrap=KAFKA_BOOTSTRAP, partitions=1, rf=1):
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        topics = await admin.list_topics()
        if name not in topics:
            await admin.create_topics([NewTopic(name=name, num_partitions=partitions, replication_factor=rf)])
            for _ in range(50):
                if name in await admin.list_topics():
                    break
                await asyncio.sleep(0.2)
    finally:
        await admin.close()

async def kafka_smoke(bootstrap=KAFKA_BOOTSTRAP):
    prod = AIOKafkaProducer(bootstrap_servers=bootstrap, value_serializer=lambda v: json.dumps(v).encode())
    await prod.start()
    await prod.stop()
    print("Kafka reachable:", bootstrap)

# --- Streaming Logic ---

def stream_slice(asset, datasets, target_scores):
    """Return exactly WINDOW + TARGET_SCORES[asset] scaled points."""
    d   = datasets[asset]
    sc  = d["scaled"]
    split = d["split"]
    need = WINDOW + target_scores[asset]
    start_idx = max(0, split - (WINDOW - 1))
    end_idx   = start_idx + need
    return sc[start_idx:end_idx]

def round_robin_payloads(datasets, target_scores):
    """Interleave per-asset sequences up to their target lengths."""
    seqs = {a: stream_slice(a, datasets, target_scores) for a in ASSETS}
    idx  = {a: 0 for a in ASSETS}
    produced = 0
    hard_cap = sum(len(v) for v in seqs.values())
    while produced < hard_cap:
        for a in ASSETS:
            if idx[a] < len(seqs[a]):
                yield a, float(seqs[a][idx[a]])
                idx[a] += 1
                produced += 1

async def producer(datasets, target_scores, bootstrap=KAFKA_BOOTSTRAP, topic=TOPIC, sleep_s=0.0):
    prod = AIOKafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode(),
        acks=0, linger_ms=0, compression_type=None
    )
    await prod.start()
    try:
        sent = 0
        for asset, val in round_robin_payloads(datasets, target_scores):
            msg = {"ts": time.time(), "asset": asset, "val": val}
            await prod.send_and_wait(topic, msg)
            sent += 1
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
        await asyncio.sleep(0.25)  # drain
        print("Producer sent:", sent)
    finally:
        await prod.stop()

async def consumer(models, scalers, datasets, target_scores, 
                  truth_buffer, pred_buffer, infer_ms_buffer, e2e_buffer,
                  bootstrap=KAFKA_BOOTSTRAP, topic=TOPIC):
    cons = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="latest",
        group_id=None,
        enable_auto_commit=False,
        client_id="mln_eval_consumer",
        fetch_max_wait_ms=1,
        fetch_min_bytes=1,
        request_timeout_ms=15000
    )
    await cons.start()
    # seek to end to ignore stale data
    while not cons.assignment():
        await asyncio.sleep(0.01)
    end_offsets = await cons.end_offsets(cons.assignment())
    for tp, off in end_offsets.items():
        cons.seek(tp, off)
    await asyncio.sleep(0.5)  # allow fetcher to settle

    # Per-asset buffers, pending predictions, and scored counters
    buf     = {a: [] for a in ASSETS}
    pending = {a: None for a in ASSETS}
    scored  = {a: 0 for a in ASSETS}

    try:
        async for msg in cons:
            recv_t = time.time()
            datum  = msg.value
            if not (isinstance(datum, dict) and "asset" in datum and "val" in datum):
                continue
            asset = datum["asset"]
            val   = float(datum["val"])
            sent_ts = float(datum.get("ts", recv_t))

            # If this asset already reached its target, ignore further ticks for it
            if scored[asset] >= target_scores[asset]:
                continue

            # finalize previous prediction for this asset with current truth
            if pending[asset] is not None:
                preds_per_model, pred_sent_ts = pending[asset]
                true_val = scalers[asset].inverse_transform([[val]])[0,0]
                truth_buffer[asset].append(true_val)

                for mname, yhat_scaled, t0, t1 in preds_per_model:
                    yhat = scalers[asset].inverse_transform([[yhat_scaled]])[0,0]
                    pred_buffer[mname][asset].append(yhat)
                    infer_ms_buffer[mname][asset].append((t1 - t0)*1000.0)

                e2e_buffer[asset].append((recv_t - pred_sent_ts)*1000.0)
                scored[asset] += 1
                pending[asset] = None

                # If ALL assets met their targets, we can stop
                if all(scored[a] >= target_scores[a] for a in ASSETS):
                    break

            # issue next prediction if window is ready and we still need predictions
            buf[asset].append(val)
            if len(buf[asset]) >= WINDOW and scored[asset] < target_scores[asset]:
                xin = torch.tensor(buf[asset][-WINDOW:], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                preds = []
                
                # Iterate over models for this asset
                for m_name, model_dict in models.items():
                    model = model_dict[asset]
                    t0 = time.time()
                    with torch.no_grad():
                        y_out = model(xin).detach().cpu().numpy().flatten()[0]
                    t1 = time.time()
                    preds.append((m_name, y_out, t0, t1))

                pending[asset] = (preds, sent_ts)

    finally:
        await cons.stop()
