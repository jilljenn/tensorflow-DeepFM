all:
	time python dfm.py --dataset last_fr_en --iter 20 --fm
	time python dfm.py --dataset last_fr_en --iter 20 --deep --fm

dummy:
	mkdir -p data/dummy
	python dummy.py
	time python dfm.py --dataset dummy --iter 1000 --rate 0.001 --batch 4 --deep --fm
	# time python dfm.py --dataset dummy --iter 100 --deep
	# time python dfm.py --dataset dummy --iter 100 --deep --fm

easy:
	time python dfm.py --dataset listen_fr_en --iter 10 --fm
	time python dfm.py --dataset listen_fr_en --iter 10 --deep
	time python dfm.py --dataset listen_fr_en --iter 10 --deep --fm

pull:
	scp raiden:deepfm/data/last_fr_en/y_pred* data/last_fr_en
	scp raiden:deepfm/data/last_es_en/y_pred* data/last_es_en
	scp raiden:deepfm/data/last_en_es/y_pred* data/last_en_es

push:
	rsync -avz --progress --partial dfm.py data *_*sh raiden:deepfm

bash:
	python makesh.py --dataset fr_en
	python makesh.py --dataset es_en
	python makesh.py --dataset en_es

clean:
	rm -r data/dummy

# awk command for posterity
# awk -F "\"*,\"*" '{print $3}' data/fren/train.csv > data/fren/y_train.csv  # 10 if fr_en
