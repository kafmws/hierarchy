import os
import sys
sys.path.append('/root/projects/readings/work')

import torch
import numpy as np
import pandas as pd

from prompts import hierarchical_prompt
from hierarchical.hierarchy import Hierarchy, get_hierarchy, Node


prompts, layer_mask, hierarchical_targets, classnames, h = hierarchical_prompt('iwildcam36')
h: Hierarchy = h

with open('/root/projects/readings/work/vis/tree_vis/h.txt', 'w') as htxt,\
    open('/root/projects/readings/work/vis/tree_vis/decorate.txt', 'w') as decoratetxt:

    roots = h.get_roots()
    # colors = ['#B0C4FF', '#FFDDBB']
    colors = ['#B0C4FF', '#F5DEB3']
    fontsizemap = [21, 12, 8, 7, 7]
    colormap = dict(zip(roots, colors))
    
    # print(colormap)

    leaf_cnt = 1
    for leaf in h.get_leaves():
        leaf: Node = leaf
        path = leaf.get_path()

        textpath = ''
        for node in path[0]:
            # if node.is_leaf():
            #     textpath += node.english_name
            # else:
            name = node.name
            name = name[0].upper() + name[1:]
            textpath += name.replace(' ', '_')
            textpath += '.'

            if not node.is_leaf():
                print(f'{name}	clade_marker_size	{int((h.n_layer - node.layer) * 30)}', file=decoratetxt) # 结点大小
                print(f'{name}	annotation_background_color	{colormap[node.get_root()]}', file=decoratetxt) # 注释背景颜色
                if node.layer < 4: # 注释名字到属级
                    print(f'{name}	annotation	{name}', file=decoratetxt)
                    print(f'{name}	annotation_font_size	{fontsizemap[node.layer]}', file=decoratetxt)
                    # print(f'{name}	annotation_rotation	90', file=decoratetxt) # 标签旋转90度
            # else:  # 显示物种级别标签
                # print(f'{name}	annotation	{leaf_cnt}:{name}', file=decoratetxt)
                # print(f'{name}	annotation	a:{name}', file=decoratetxt)
                
        print(textpath[:-1], file=htxt)

os.chdir('/root/projects/readings/work/vis/tree_vis')
os.system('graphlan_annotate.py --annot sketch.txt     h.txt       h0.xml')
os.system('graphlan_annotate.py --annot decorate.txt      h0.xml     h1.xml')
os.system('graphlan.py h1.xml h.png --dpi 300 --size 9')
