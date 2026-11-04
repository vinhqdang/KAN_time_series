// Monte Carlo of the Galton-Watson (Poisson(c)) local limit around a uniform edge (i,j):
// estimates P(Psi) by directly sampling the definition (no closed form used), compares with F_up.
#include <bits/stdc++.h>
using namespace std;
mt19937_64 rng(12345); double C; long long CAPV=3000;
int pois(){ static poisson_distribution<int> pd; return 0; }
poisson_distribution<int>* PD;
inline int P(){ return (*PD)(rng); }
bool survives(){ // GW Poisson(C) from one root; infinite iff total size exceeds cap
  long long cur=1,total=1; const long long cap=CAPV;
  while(cur>0){ long long nxt=0; for(long long t=0;t<cur;t++){ nxt+=P(); if(nxt>cap) return true; } total+=nxt; if(total>cap) return true; cur=nxt; }
  return false; }
bool allOtherFinite(){ int n=P(); for(int t=0;t<n;t++) if(survives()) return false; return true; }
bool Q(int depth){ // v has indegree 0, or a parent u with all other children finite and Q(u)
  int n=P(); if(n==0) return true; if(depth>=400) return false;
  for(int t=0;t<n;t++){ if(allOtherFinite() && Q(depth+1)) return true; }
  return false; }
bool inTreeInfinite(){ return survives(); } // ancestor tree is also GW Poisson(C)
int main(int argc,char**argv){
  C=atof(argv[1]); long long n=atoll(argv[2]); if(argc>3) CAPV=atoll(argv[3]); PD=new poisson_distribution<int>(C);
  long long a=0,c1=0,c2=0,A=0,Dj=0; long long both=0;
  for(long long s=0;s<n;s++){
    bool Ai=inTreeInfinite(); bool Dji=survives(); int nk=P();
    bool cl1=(!Ai && nk>=1);
    bool cl2=false; if(!Dji){ for(int t=0;t<nk;t++){ if(allOtherFinite() && Q(0)){cl2=true;break;} } }
    A+=Ai; Dj+=Dji; c1+=cl1; c2+=cl2; a+= (cl1||cl2); both+=(cl1&&cl2);
  }
  printf("c=%g n=%lld  P(A(i))=%.5f P(D(j))=%.5f P(clause1)=%.5f P(clause2)=%.5f P(Psi)=%.5f (se %.5f) P(c1&c2)=%.5f\n",C,n,(double)A/n,(double)Dj/n,(double)c1/n,(double)c2/n,(double)a/n,sqrt((double)a/n*(1-(double)a/n)/n),(double)both/n);
}
