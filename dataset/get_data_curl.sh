
mkdir images
cd images
for V in train2017 val2017 test2017
do
	echo Getting $V

	wget --timeout 10 http://images.cocodataset.org/zips/$V.zip
	unzip -q $V.zip
	rm $V.zip
	
done
cd ..

mkdir /annotations
cd /annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
cd ..

echo Done

