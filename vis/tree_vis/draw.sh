cd /root/projects/readings/work/vis/tree_vis
python draw.py
graphlan_annotate.py --annot sketch.txt     h.txt       h0.xml
graphlan_annotate.py --annot decorate.txt      h0.xml     h1.xml
graphlan.py h1.xml h.png --dpi 300 --size 6