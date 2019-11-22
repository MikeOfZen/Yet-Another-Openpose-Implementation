mkdir images

echo getting train2017
gsutil -m rsync gs://images.cocodataset.org/train2017 ./images/
echo getting val2017
gsutil -m rsync gs://images.cocodataset.org/val2017 ./images/
echo getting test2017
gsutil -m rsync gs://images.cocodataset.org/test2017 ./images/

mkdir /annotations
cd /annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
unzip annotations_trainval2017.zip

echo Done

