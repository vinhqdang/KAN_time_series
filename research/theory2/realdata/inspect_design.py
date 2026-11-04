"""Stage A design inspection (no correlations): for each Causal Chambers walk file, the number of rows,
the sampling interval, and how many distinct values each experimenter-set input takes (to know which inputs
are random variables in that experiment).  Usage: python3 -I inspect_design.py <dir_with_csvs>..."""
import sys, os, glob, csv
import numpy as np
LT_IN = ["red","green","blue","osr_c","v_c","pol_1","pol_2","osr_angle_1","osr_angle_2","v_angle_1","v_angle_2",
         "l_11","l_12","l_21","l_22","l_31","l_32","diode_ir_1","diode_vis_1","diode_ir_2","diode_vis_2","diode_ir_3",
         "diode_vis_3","t_ir_1","t_vis_1","t_ir_2","t_vis_2","t_ir_3","t_vis_3"]
WT_IN = ["hatch","pot_1","pot_2","osr_1","osr_2","osr_mic","osr_in","osr_out","osr_upwind","osr_downwind","osr_ambient",
         "osr_intake","v_1","v_2","v_mic","v_in","v_out","load_in","load_out","res_in","res_out"]
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        with open(f) as fh:
            r = csv.reader(fh); head = next(r); rows = list(r)
        cols = {h: i for i, h in enumerate(head)}
        ins = LT_IN if "red" in cols else WT_IN
        ts = np.array([float(x[cols["timestamp"]]) for x in rows])
        dt = np.median(np.diff(ts))
        var = []
        for c in ins:
            vals = set(x[cols[c]] for x in rows)
            if len(vals) > 1: var.append(f"{c}:{len(vals)}")
        flags = set(x[cols["flag"]] for x in rows); ivs = set(x[cols["intervention"]] for x in rows)
        print(f"{os.path.basename(f)} n={len(rows)} dt_med={dt:.4f}s flags={sorted(flags)[:4]} intervention_vals={sorted(ivs)[:4]} varying_inputs=[{', '.join(var)}]")
