mkdir images
cd images

for V in train2017 val2017 test2017
do
	echo Getting $V

	curl http://images.cocodataset.org/zips/$V.zip -o $V.zip
	echo Unziping $V
	unzip -qo $V.zip
	echo Deleting $V
	rm $V.zip
done
cd ..

mkdir /annotations
cd /annotations
curl http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
cd ..

echo Done

