clc;
clear;
close all;
warning off;

className_ = {'spam','ham'};

% ###### read english big data 
f1_ = fopen('english_big.txt');
count_ = 0;
while true
    l_ = fgetl(f1_);
    if ~ischar(l_)
       break;
    end
    commaIndex_ = strfind(l_,',');commaIndex_ = commaIndex_(end);
    count_ = count_ + 1;
    englishBig_{count_} = l_(1:commaIndex_-1);
    englishBigLabel_{count_} = l_(commaIndex_+1:end);
end
fclose(f1_);

% convert char label to numeric label
englishBigLabelNumeric_ = zeros(length(englishBigLabel_),1);
for class_ = 1 : length(className_)
    commaIndex_ = strcmp(englishBigLabel_,className_(class_));
    englishBigLabelNumeric_(commaIndex_) = class_;
end

% process messages
englishBigBag_ = erasePunctuation(englishBig_);
englishBigBag_ = lower(englishBigBag_);
englishBigBag_ = tokenizedDocument(englishBigBag_);
englishBigBag_ = removeWords(englishBigBag_,stopWords);
englishBigBag_ = removeShortWords(englishBigBag_,3);
englishBigBag_ = removeLongWords(englishBigBag_,10);
englishBigBag_ = normalizeWords(englishBigBag_);
englishBigBag_ = removeWords(englishBigBag_,stopWords);
englishBigBag_ = bagOfWords(englishBigBag_);
englishBigBag_ = removeInfrequentWords(englishBigBag_,10);
[englishBigBag_,empty_] = removeEmptyDocuments(englishBigBag_);
englishBigLabelNumeric_(empty_) = [];
englishBig_ = full(englishBigBag_.tfidf);

% ###### randperm data
perm_ = randperm(size(englishBig_,1));
englishBig_ = englishBig_(perm_,:);
englishBigLabelNumeric_ = englishBigLabelNumeric_(perm_);

% ###### splite data
trainCount_ = round(0.5 * size(englishBig_,1));
X1_ = englishBig_(1:trainCount_,:);
Y1_ = englishBigLabelNumeric_(1:trainCount_);
X2_ = englishBig_(1+trainCount_:end,:);
Y2_ = englishBigLabelNumeric_(1+trainCount_:end);

trainCount_ = round(0.75 * size(X1_,1));
X3_ = X1_(1:trainCount_,:);
Y3_ = Y1_(1:trainCount_);
X4_ = X1_(1+trainCount_:end,:);
Y4_ = Y1_(1+trainCount_:end);


% ###### training
[X1Eval_] = classificaitonMehtod(X1_,Y1_);
[X2Eval_] = classificaitonMehtod(X2_,Y2_);
[X3Eval_] = classificaitonMehtod(X3_,Y3_);
[X4Eval_] = classificaitonMehtod(X4_,Y4_);

% ###### evaluating
% macroF1Measure
precison1_ = X1Eval_(:,6);
precison2_ = X2Eval_(:,6);
recall1_ = X1Eval_(:,7);
recall2_ = X2Eval_(:,7);
macPrecision_ = (precison1_ + precison2_)./2;
macRecall_ = (recall1_+recall2_)./2;
macF1Measure_ = ( 2.* ((macPrecision_.*macRecall_)./(macPrecision_+macRecall_)) )';

% microF1Measure
TP1_ = X3Eval_(:,1);
TP2_ = X4Eval_(:,1);
FP1_ = X3Eval_(:,3);
FP2_ = X4Eval_(:,3);
FN1_ = X3Eval_(:,4);
FN2_ = X4Eval_(:,4);
micPrecision_=((TP1_+TP2_)./(TP1_+TP2_+FP1_+FP2_));
micRecall_=((TP1_+TP2_)./(TP1_+TP2_+FN1_+FN2_));
micF1Measure_ =  2*((micPrecision_.*micRecall_)./(micPrecision_+micRecall_) )';

% accuracy
accuracy_ = mean([X1Eval_(:,5) X2Eval_(:,5) X3Eval_(:,5) X4Eval_(:,5)]');

% ###### show results
table_ = array2table([accuracy_;micF1Measure_;macF1Measure_]','VariableNames',{'accuracy','microF1Measure','macroF1Measure'} ...
    ,'RowNames',{'bagger','decisionTree','knn','naiveBayes'});
disp(table_);
