
 cd /root/workspace/YOLO-Master
python /root/workspace/YOLO-Master-docs/issue2/VisDrone/phase2-pruned/scripts/step1_export_usage.py


cd /root/workspace/YOLO-Master
  python /root/workspace/YOLO-Master-docs/issue2/VisDrone/phase2-pruned/scripts/st
  ep2_prune.py --mode usage --threshold 0.15

  单组跑通后再全量 10 组：

  python /root/workspace/YOLO-Master-docs/issue2/VisDrone/phase2-pruned/scripts/st
  ep2_prune.py
