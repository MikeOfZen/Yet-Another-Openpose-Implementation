mkdir train2017
mkdir val2017
mkdir test2017

echo getting train2017
gsutil -m rsync gs://images.cocodataset.org/train2017 $BASH_SOURCE/train2017
echo getting val2017
gsutil -m rsync gs://images.cocodataset.org/val2017 $BASH_SOURCE/val2017
echo getting test2017
gsutil -m rsync gs://images.cocodataset.org/test2017 $BASH_SOURCE/test2017

mkdir $BASH_SOURCE/annotations
cd $BASH_SOURCE/annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
unzip annotations_trainval2017.zip

echo Done

