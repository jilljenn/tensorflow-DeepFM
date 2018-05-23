all:
	time python test.py --dataset last_fr_en --iter 20 --fm
	time python test.py --dataset last_fr_en --iter 20 --deep --fm

dummy:
	time python test.py --dataset dummy --iter 100 --fm
	time python test.py --dataset dummy --iter 100 --deep
	time python test.py --dataset dummy --iter 100 --deep --fm

easy:
	time python test.py --dataset listen_fr_en --iter 10 --fm
	time python test.py --dataset listen_fr_en --iter 10 --deep
	time python test.py --dataset listen_fr_en --iter 10 --deep --fm

pull:
	scp raiden:deepfm/data/last_fr_en/*.txt data/last_fr_en
	scp raiden:deepfm/data/last_es_en/*.txt data/last_es_en
	scp raiden:deepfm/data/last_en_es/*.txt data/last_en_es

push:
	rsync -avz --progress --partial test.py data *_*sh raiden:deepfm

bash:
	python makesh.py
	python makesh.py --iter 200

clean:
	rm data/dummy/*

# awk command for posterity
# awk -F "\"*,\"*" '{print $3}' data/fren/train.csv > data/fren/y_train.csv  # 10 if fr_en
