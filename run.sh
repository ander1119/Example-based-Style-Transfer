
## Get G_seg
python3 ./code/Gseg/segment.py  --input_dir ./inputImage/styleImage  --conf_threshold 0.99
python3 ./code/Gseg/segment.py  --input_dir ./inputImage/targetImage --conf_threshold 0.99


## Get G_pos
python3 code/Gpos/mlsAffineMorpher.py ./inputImage/styleImage style1.png ./inputImage/targetImage target1.png

## Get G_app
python3 code/Gapp/generateGapp.py  ./inputImage/styleImage style1.png ./inputImage/targetImage target1.png