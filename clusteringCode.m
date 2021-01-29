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
trainCount_ = round(0.7 * size(englishBig_,1));
Xtr_ = englishBig_(1:trainCount_,:);
Ytr_ = englishBigLabelNumeric_(1:trainCount_);
Xts_ = englishBig_(1+trainCount_:end,:);
Yts_ = englishBigLabelNumeric_(1+trainCount_:end);

% ###### cluster data
[clusters_,centroids_]  = kmeans(Xtr_,2);
if mode(clusters_) ~= mode(Ytr_)
    temp_ = centroids_(1,:);
    centroids_(1,:) = centroids_(2,:);
    centroids_(2,:) = temp_;
end
for i = 1 : 2
    ED_(i,:) = sqrt(sum((Xts_-centroids_(i,:)).^2,2));
end
[~,predictLables_] = min(ED_);
clusEval_ = EvalCrit(predictLables_',Yts_);
clusEval_ = [clusEval_(5),clusEval_(6),clusEval_(7)];

% ###### show results
figure;
bar(clusEval_,0.5);
set(gca,'XTickLabel',{'accuracy';'precision';'recall'});
text(1:length(clusEval_),clusEval_,num2str(clusEval_'),'vert','bottom','horiz','center');

figure;
plotconfusion(ind2vec(predictLables_),ind2vec(Yts_'));
title('confusion from clustering');

% ###### show results
model_ = fitctree(Xtr_,Ytr_);
predictLables_ = predict(model_,Xts_);
figure;
plotconfusion(ind2vec(predictLables_'),ind2vec(Yts_'));
title('confusion from decision tree');