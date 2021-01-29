function output_ = EvalCrit(Yp_,Y_)
    confusionMatrix_ = confusionmat(Y_,Yp_);
    TP_ = confusionMatrix_(1,1);
    TN_ = confusionMatrix_(2,2);
    FP_ = confusionMatrix_(2,1);
    FN_ = confusionMatrix_(1,2);
           
    recall_ = (TP_ / (TP_ + FN_));
    precision_ = TP_ / (TP_ + FP_);
    accuracy_ = (TP_ + TN_) / sum(confusionMatrix_(:));
        
    output_ = [TP_,TN_,FP_,FN_,accuracy_,precision_,recall_];
end