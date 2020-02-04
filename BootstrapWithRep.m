function [BootSample_pos,BootSample_neg,BootSample_pn]=BootstrapWithRep(no_node,NB,no_train,train_vcts,train_behav,thresh)

% This function performs bootstrapping with replacement as feature selection.

% Input

% 'no_node'           number of nodes (or ROIs)
% 'NB'                time of resampling
% 'no_train'          number of training subjects
% 'train_vcts'        trainning data in a matrix of size (number of features, number of training subjects)
% 'train_behav'       behaviral data in a vector of size (number of training subjects, 1)
% 'thresh'            P threshold of correlation test

% Output

% 'BootSample_pos'    number of time each feature is selected as posivitively correlated feature
% 'BootSample_neg'    number of time each feature is selected as negatively correlated feature
% 'BootSample_pn'     number of time each feature is selected as correlated feature


% initialization

% BootSample_pos=zeros(no_node,no_node);
% BootSample_neg=zeros(no_node,no_node);
% BootSample_pn=zeros(no_node,no_node);
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
    
    subject_Index=randsample(no_train,no_train,true);
    sub_train_x=train_vcts(:,subject_Index);
    sub_train_y=train_behav(subject_Index);
    
    % correlate all features with behavior
    
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
