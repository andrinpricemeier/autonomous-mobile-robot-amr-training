RUN_NAME=yolov5s_$CI_COMMIT_SHA
rm -fr /usr/local/stairway-jones/jetson-trainer
mkdir /usr/local/stairway-jones/jetson-trainer
cp -R yolov5-v4.0 /usr/local/stairway-jones/jetson-trainer/
rm -fr /usr/local/stairway-jones/data
cp -R data /usr/local/stairway-jones/
rm -fr /usr/local/stairway-jones/preparation
cp -R preparation /usr/local/stairway-jones/
mkdir /usr/local/stairway-jones/data/augmented_jetson
mkdir /usr/local/stairway-jones/data/augmented_mobile_phone
cd /usr/local/stairway-jones/preparation
python3 pipeline.py -i "/usr/local/stairway-jones/data/annotated_mobile_phone" -o "/usr/local/stairway-jones/data/augmented_mobile_phone" -r 0.0 -p "mobile-phone"
python3 pipeline.py -i "/usr/local/stairway-jones/data/annotated_jetson" -o "/usr/local/stairway-jones/data/augmented_jetson" -p "jetson"
cp -r /usr/local/stairway-jones/data/augmented_mobile_phone/yolo/images/train/. /usr/local/stairway-jones/data/augmented_jetson/yolo/images/train/
cp -r /usr/local/stairway-jones/data/augmented_mobile_phone/yolo/labels/train/. /usr/local/stairway-jones/data/augmented_jetson/yolo/labels/train/
cp -R /usr/local/stairway-jones/data/augmented_jetson/yolo /usr/local/stairway-jones/jetson-trainer/data
echo "python3 train.py --batch 16 --img 640 --epochs 300 --data data.yaml --cfg yolov5s.yaml --weights 'yolov5s.pt' --name $RUN_NAME --hyp hyp.finetune.yaml" > /usr/local/stairway-jones/jetson-trainer/yolov5-v4.0/train.sh
cd /usr/local/stairway-jones
rm -fr jetson_trainer.zip
zip -r jetson_trainer.zip jetson-trainer