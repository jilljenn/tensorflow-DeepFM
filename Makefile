2019:
	# time python fm.py --dataset duolingo2019_fr_en --iter 200
	# time python fm.py --dataset duolingo2019_fr_en --ver 3 --logistic
	time python fm.py --dataset duolingo2019_fr_en --ver 3 --iter 100 --d 1
	# time python fm.py --dataset duolingo2019_fr_en --ver 3 --iter 100 --d 2
	# time python fm.py --dataset simplest_fr_en --logistic
	# time python fm.py --dataset simplest_fr_en --logistic --countries
	# time python fm.py --dataset simplest_fr_en --iter 20

all:
	time python dfm.py --dataset last_fr_en --iter 20 --fm
	time python dfm.py --dataset last_fr_en --iter 20 --deep --fm

eval:
	python eval.py
	cp results* /Users/jilljenn/code/ktm/poster/tables

dummy:
	mkdir -p data/dummy
	python dummy.py
	time python fm.py --dataset dummy --logistic
	time python fm.py --dataset dummy
	time python dfm.py --dataset dummy --iter 1000 --rate 0.001 --batch 4 --deep --fm
	# time python dfm.py --dataset dummy --iter 100 --deep
	# time python dfm.py --dataset dummy --iter 100 --deep --fm

easy:
	time python dfm.py --dataset listen_fr_en --iter 10 --fm
	time python dfm.py --dataset listen_fr_en --iter 10 --deep
	time python dfm.py --dataset listen_fr_en --iter 10 --deep --fm

pull:
	# rsync -avz raiden:deepfm/data/first_fr_en/y_pred* data/first_fr_en
	# rsync -avz raiden:deepfm/data/first_es_en/y_pred* data/first_es_en
	# rsync -avz raiden:deepfm/data/first_en_es/y_pred* data/first_en_es
	# rsync -avz raiden:deepfm/data/last_fr_en/y_pred* data/last_fr_en
	# rsync -avz raiden:deepfm/data/last_es_en/y_pred* data/last_es_en
	# rsync -avz raiden:deepfm/data/last_en_es/y_pred* data/last_en_es
	rsync -avz raiden:deepfm/data/pfa_fr_en/y_pred* data/pfa_fr_en
	rsync -avz raiden:deepfm/data/pfa_es_en/y_pred* data/pfa_es_en
	rsync -avz raiden:deepfm/data/pfa_en_es/y_pred* data/pfa_en_es

push:
	rsync -avz --progress --partial requirements.txt Makefile fm.py dfm.py data *_*sh raiden:deepfm

bash:
	python makesh.py --dataset fr_en
	python makesh.py --dataset es_en
	python makesh.py --dataset en_es

clean:
	rm -r data/dummy

# awk command for posterity
# awk -F "\"*,\"*" '{print $3}' data/fren/train.csv > data/fren/y_train.csv  # 10 if fr_en
