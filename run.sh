
## Get G_seg
# python3 ./code/Gseg/segment.py  --input_dir ./inputImage/styleImage  --conf_threshold 0.99
# python3 ./code/Gseg/segment.py  --input_dir ./inputImage/targetImage --conf_threshold 0.99

## Get G_pos
python3 code/Gpos/mlsAffineMorpher.py ./inputImage/styleImage $1.png ./inputImage/targetImage $2.png

## Get G_app
python3 code/Gapp/generateGapp.py  ./inputImage/styleImage $1.png ./inputImage/targetImage $2.png