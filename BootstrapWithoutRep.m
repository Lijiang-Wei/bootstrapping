function [BootSample_pos,BootSample_neg,BootSample_pn]=BootstrapWithoutRep(NB,no_train,train_vcts,train_behav,thresh,BootPer)

% This function performs bootstrapping without replacement as feature selection.

% Input

% 'NB'                time of resampling
% 'no_train'          number of training subjects
% 'train_vcts'        trainning data in a matrix of size (number of features, number of training subjects)
% 'train_behav'       behaviral data in a vector of size (number of training subjects, 1)
% 'thresh'            P threshold of correlation test
% 'BootPer'           resampling percentage, i.e., the ratio of resampled samples to original samples

% Output

% 'BootSample_pos'    number of time each feature is selected as posivitively correlated feature
% 'BootSample_neg'    number of time each feature is selected as negatively correlated feature
% 'BootSample_pn'     number of time each feature is selected as correlated feature


% initialization

BootSample_pos=zeros(size(train_vcts,1),1);
BootSample_neg=zeros(size(train_vcts,1),1);
BootSample_pn=zeros(size(train_vcts,1),1);

idx_pos=cell(1,NB);
idx_neg=cell(1,NB);
idx_pn=cell(1,NB);

% sample from original dataset

rng('shuffle')
parfor nboot=1:NB

    % implement on one subset
    
    subject_Index=randsample(no_train,round(no_train*BootPer));
    sub_train_x=train_vcts(:,subject_Index);
    sub_train_y=train_behav(subject_Index);
    
    % correlate all features with behavioral measure
    
    [RHO,PVAL]=corr(sub_train_x',sub_train_y,'type','Spearman');
    
    idx_pos{nboot}=find(PVAL<thresh & RHO>0);
    idx_neg{nboot}=find(PVAL<thresh & RHO<0);
    idx_pn{nboot}=find(PVAL<thresh);
end

% counting

for nboot2=1:NB
    BootSample_pos(idx_pos{nboot2})=BootSample_pos(idx_pos{nboot2})+1;
    BootSample_neg(idx_neg{nboot2})=BootSample_neg(idx_neg{nboot2})+1;
    BootSample_pn(idx_pn{nboot2})=BootSample_pn(idx_pn{nboot2})+1;
end

end
