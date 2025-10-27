export PYTHONPATH=/home/muratbal/glue-factory/LightGlue:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,0
python -m gluefactory.train aliked_twin_lg_homography --conf gluefactory/configs/aliked_twin+lightglue_homography.yaml  --restore
