#!/bin/bash
PERCENTAGE_PRIVILEGED="5 10 25 30 50"
N=100
cd data
for f in *.mat;
do
	mkdir ../result_$(date +'%d_%m_%Y')
	cd ../result_$(date +'%d_%m_%Y')
	mkdir ${f%.mat}
	for i in ${PERCENTAGE_PRIVILEGED}
	do
		mkdir ${f%.*}/privileged_$i
		cd ${f%.*}/privileged_$i
		for j in {1..100}
		do
			mkdir run_$j
			cd run_$j
			mkdir saved_models
			mkdir saved_images
			python ../../../../architecture_input.py $f $i $j >> res.txt
			
			while IFS=" " read -r train_m1_acc train_m2_acc train_m3_acc train_m1_loss train_m2_loss train_m3_loss train_total_loss test_m1_acc test_m2_acc test_m3_acc test_total_acc test_m1_loss test_m2_loss test_m3_loss test_total_loss crossentropy_final precision_m1 precision_m2 precision_m3 precision_final recall_m1 recall_m2 recall_m3 recall_final fbeta_m1 fbeta_m2 fbeta_m3 fbeta_final 
			do
				echo " "
			done < res.txt
			
			tr_m1_acc=$(awk "BEGIN {print $tr_m1_acc+$train_m1_acc}")
			tr_m2_acc=$(awk "BEGIN {print $tr_m2_acc+$train_m2_acc}")
			tr_m3_acc=$(awk "BEGIN {print $tr_m3_acc+$train_m3_acc}")
			tr_m1_loss=$(awk "BEGIN {print $tr_m1_loss+$train_m1_loss}")
			tr_m2_loss=$(awk "BEGIN {print $tr_m2_loss+$train_m2_loss}")
			tr_m3_loss=$(awk "BEGIN {print $tr_m3_loss+$train_m3_loss}")
			tr_total_loss=$(awk "BEGIN {print $tr_total_loss+$train_total_loss}")
			te_m1_acc=$(awk "BEGIN {print $te_m1_acc+$test_m1_acc}")
			te_m2_acc=$(awk "BEGIN {print $te_m2_acc+$test_m2_acc}")
			te_m3_acc=$(awk "BEGIN {print $te_m3_acc+$test_m3_acc}")
			te_total_acc=$(awk "BEGIN {print $te_total_acc+$test_total_acc}")
			te_m1_loss=$(awk "BEGIN {print $te_m1_loss+$test_m1_loss}")
			te_m2_loss=$(awk "BEGIN {print $te_m2_loss+$test_m2_loss}")
			te_m3_loss=$(awk "BEGIN {print $te_m3_loss+$test_m3_loss}")
			te_total_loss=$(awk "BEGIN {print $te_total_loss+$test_total_loss}")
			ce_final=$(awk "BEGIN {print $ce_final+$crossentropy_final}")
			pre_m1=$(awk "BEGIN {print $pre_m1+$precision_m1}")
			pre_m2=$(awk "BEGIN {print $pre_m2+$precision_m2}")
			pre_m3=$(awk "BEGIN {print $pre_m3+$precision_m3}")
			pre_final=$(awk "BEGIN {print $pre_final+$precision_final}")
			re_m1=$(awk "BEGIN {print $re_m1+$recall_m1}")
			re_m2=$(awk "BEGIN {print $re_m2+$recall_m2}")
			re_m3=$(awk "BEGIN {print $re_m3+$recall_m3}")
			re_final=$(awk "BEGIN {print $re_final+$recall_final}")
			fb_m1=$(awk "BEGIN {print $fb_m1+$fbeta_m1}")
			fb_m2=$(awk "BEGIN {print $fb_m2+$fbeta_m2}")
			fb_m3=$(awk "BEGIN {print $fb_m3+$fbeta_m3}")
			fb_final=$(awk "BEGIN {print $fb_final+$fbeta_final}")
			
			cd ..
		done
				
		tr_m1_acc=$(awk "BEGIN {print $tr_m1_acc/$N}")
		tr_m2_acc=$(awk "BEGIN {print $tr_m2_acc/$N}")
		tr_m3_acc=$(awk "BEGIN {print $tr_m3_acc/$N}")
		tr_m1_loss=$(awk "BEGIN {print $tr_m1_loss/$N}")
		tr_m2_loss=$(awk "BEGIN {print $tr_m2_loss/$N}")
		tr_m3_loss=$(awk "BEGIN {print $tr_m3_loss/$N}")
		tr_total_loss=$(awk "BEGIN {print $tr_total_loss/$N}")
		te_m1_acc=$(awk "BEGIN {print $te_m1_acc/$N}")
		te_m2_acc=$(awk "BEGIN {print $te_m2_acc/$N}")
		te_m3_acc=$(awk "BEGIN {print $te_m3_acc/$N}")
		te_total_acc=$(awk "BEGIN {print $te_total_acc/$N}")
		te_m1_loss=$(awk "BEGIN {print $te_m1_loss/$N}")
		te_m2_loss=$(awk "BEGIN {print $te_m2_loss/$N}")
		te_m3_loss=$(awk "BEGIN {print $te_m3_loss/$N}")
		te_total_loss=$(awk "BEGIN {print $te_total_loss/$N}")
		ce_final=$(awk "BEGIN {print $ce_final/$N}")
		pre_m1=$(awk "BEGIN {print $pre_m1/$N}")
		pre_m2=$(awk "BEGIN {print $pre_m2/$N}")
		pre_m3=$(awk "BEGIN {print $pre_m3/$N}")
		pre_final=$(awk "BEGIN {print $pre_final/$N}")
		re_m1=$(awk "BEGIN {print $re_m1/$N}")
		re_m2=$(awk "BEGIN {print $re_m2/$N}")
		re_m3=$(awk "BEGIN {print $re_m3/$N}")
		re_final=$(awk "BEGIN {print $re_final/$N}")
		fb_m1=$(awk "BEGIN {print $fb_m1/$N}")
		fb_m2=$(awk "BEGIN {print $fb_m2/$N}")
		fb_m3=$(awk "BEGIN {print $fb_m3/$N}")
		fb_final=$(awk "BEGIN {print $fb_final/$N}")
		
		echo "priv_$i $tr_m1_acc $tr_m2_acc $tr_m3_acc $tr_m1_loss $tr_m2_loss $tr_m3_loss $tr_total_loss $te_m1_acc $te_m2_acc $te_m3_acc $te_total_acc $te_m1_loss $te_m2_loss $te_m3_loss $te_total_loss $ce_final $pre_m1 $pre_m2 $pre_m3 $pre_final $re_m1 $re_m2 $re_m3 $re_final $fb_m1 $fb_m2 $fb_m3 $fb_final" >> "../final_result_${f%.*}.csv"
		
		tr_m1_acc=0
		tr_m2_acc=0
		tr_m3_acc=0
		tr_m1_loss=0
		tr_m2_loss=0
		tr_m3_loss=0
		tr_total_loss=0
		te_m1_acc=0
		te_m2_acc=0
		te_m3_acc=0
		te_total_acc=0
		te_m1_loss=0
		te_m2_loss=0
		te_m3_loss=0
		te_total_loss=0
		ce_final=0
		pre_m1=0
		pre_m2=0
		pre_m3=0
		pre_final=0
		re_m1=0
		re_m2=0
		re_m3=0
		re_final=0
		fb_m1=0
		fb_m2=0
		fb_m3=0
		fb_final=0
		
		cd ../../
	done
done




