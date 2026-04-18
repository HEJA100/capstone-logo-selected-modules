#!/usr/bin/env bash
set -euo pipefail

PROM=/home/users/nus/e1538285/scratch/LOGO/02_LOGO_Promoter

jid1=$(qsub "$PROM/pbs_seq_multi.sh")
jid2=$(qsub "$PROM/pbs_seq_single.sh")
jid3=$(qsub "$PROM/pbs_seq_depthwise.sh")
jid4=$(qsub "$PROM/pbs_structural_none.sh")
jid5=$(qsub "$PROM/pbs_structural_single.sh")
jid6=$(qsub "$PROM/pbs_structural_depthwise.sh")

echo "submitted:"
echo "  $jid1  seq_multi"
echo "  $jid2  seq_single"
echo "  $jid3  seq_depthwise"
echo "  $jid4  structural_none"
echo "  $jid5  structural_single"
echo "  $jid6  structural_depthwise"
