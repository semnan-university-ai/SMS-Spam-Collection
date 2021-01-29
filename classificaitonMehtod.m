function [EcTS_] = classificaitonMehtod(inputData_,inputLabels_)
    disp('training ...');

    trainCount_ = round(0.75 * size(inputData_,1));
    trainData_ = inputData_(1:trainCount_,:);
    trainLabels_ = inputLabels_(1:trainCount_);
    testData_ = inputData_(trainCount_+1:end,:);
    testLabels_ = inputLabels_(trainCount_+1:end);
   
    % bagger
    model_ = TreeBagger(100,trainData_,trainLabels_,'Method','classification');    
    modelLabelsTS_ = predict(model_, testData_);
    modelLabelsTS_ = str2double(modelLabelsTS_);
    EcTS_(1,:) = EvalCrit(modelLabelsTS_,testLabels_);

    % tree
    model_ = fitctree(trainData_,trainLabels_);
    modelLabelsTS_ = predict(model_, testData_);
    EcTS_(2,:) = EvalCrit(modelLabelsTS_,testLabels_);
    
    % knn
    model_ = fitcknn(trainData_,trainLabels_,'NumNeighbors',5);
    modelLabelsTS_ = predict(model_, testData_);
    EcTS_(3,:) = EvalCrit(modelLabelsTS_,testLabels_);

    % naive bayes
    model_ = fitcnb(trainData_,trainLabels_,'DistributionNames','mvmn');
    modelLabelsTS_ = predict(model_, testData_);
    EcTS_(4,:) = EvalCrit(modelLabelsTS_,testLabels_);
end