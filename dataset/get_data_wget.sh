
for V in train2017 val2017 test2017
do
	echo Getting $V
	mkdir $V
	cd $V
	wget http://images.cocodataset.org/zips/$V.zip
	unzip $V.zip
	cd ..
done

mkdir /annotations
cd /annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
unzip annotations_trainval2017.zip
cd ..

echo Done

