function [feature_pos,feature_neg,feature_pn]=BootstrapWithRep(no_node,NB,no_train,train_mats,train_behav,P_threshold)

% initialization

sum_pos=zeros(no_node,no_node);
sum_neg=zeros(no_node,no_node);
sum_pn=zeros(no_node,no_node);

index_pos=cell(1,NB);
index_neg=cell(1,NB);
index_pn=cell(1,NB);

% sample from original dataset

rng('shuffle')
[~,NewSample] = bootstrp(NB,[],[1:no_train]);

parfor nboot=1:NB 
    
    % implement on one subset
    
    subject_Index=NewSample(:,nboot);
    sub_train_vcts=train_mats(:,subject_Index);
    sub_train_behav=train_behav(subject_Index);
    
    % correlate all features with behavior
    
    [r_vcts,p_vcts]=corr(sub_train_vcts',sub_train_behav,'type','Spearman');
    
    index_pos{nboot}=find(p_vcts<P_threshold & r_vcts>=0);
    index_neg{nboot}=find(p_vcts<P_threshold & r_vcts<0);
    index_pn{nboot}=find(p_vcts<P_threshold);
    
end
% counting
for nboot2=1:NB
    sum_pos(index_pos{nboot2})=sum_pos(index_pos{nboot2})+1;
    sum_neg(index_neg{nboot2})=sum_neg(index_neg{nboot2})+1;
    sum_pn(index_pn{nboot2})=sum_pn(index_pn{nboot2})+1;
end

% select stable features with frequency more than FP

feature_pos=find(sum_pos>=FP*NB);
feature_neg=find(sum_neg>=FP*NB);
feature_pn=find(sum_pn>=FP*NB);
end
