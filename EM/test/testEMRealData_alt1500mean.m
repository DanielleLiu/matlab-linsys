%%
addpath(genpath('../aux/'))
addpath(genpath('../kalman/'))
addpath(genpath('../data/'))
addpath(genpath('../sPCA/'))
addpath(genpath('../../robustCov/'))
%%
clear all
%% Load real data:
load ../data/dynamicsData.mat
addpath(genpath('./fun/'))
% Some pre-proc
B=nanmean(allDataEMG{1}(end-45:end-5,:,:)); %Baseline: last 40, exempting 5
clear data dataSym
subjIdx=2:16;
%muscPhaseIdx=[1:(180-24),(180-11:180)];
%muscPhaseIdx=[muscPhaseIdx,muscPhaseIdx+180]; %Excluding PER
muscPhaseIdx=1:360;
for i=1:3 %B,A,P
    %Remove baseline
    data{i}=allDataEMG{i}-B;

    %Interpolate over NaNs %This is only needed if we want to run fast
    %estimations, or if we want to avoid all subjects' data at one
    %timepoint from being discarded because of a single subject's missing
    %data
    for j=1:size(data{i},3) %each subj
       t=1:size(data{i},1); nanidx=any(isnan(data{i}(:,:,j)),2); %Any muscle missing
       data{i}(:,:,j)=interp1(t(~nanidx),data{i}(~nanidx,:,j),t,'linear',0); %Substitute nans
    end

    %Two subjects have less than 600 Post strides: C06, C08
    %Option 1: fill with zeros (current)
    %Option 2: remove them
    %Option 3: Use only 400 strides of POST, as those are common to all
    %subjects

    %Remove subj:
    data{i}=data{i}(:,muscPhaseIdx,subjIdx);

    %Compute asymmetry component
    aux=data{i}-fftshift(data{i},2); %slow- fast leg.
    dataSym{i}=aux(:,1:size(aux,2)/2,:);

end
%% Initialize the labels to load data.
%dataSym: {1-3} B,A,P; 3D: strides,muscIndex, subjectIndex
%1-1350, taken from the 2nd baseline last 50 strides (out of 150), 900 long ada, 400
%post (out of 600)
%start from trial index 5
mOrder={'TA', 'PER', 'SOL', 'LG', 'MG', 'BF', 'SEMB', 'SEMT', 'VM', 'VL', 'RF', 'HIP', 'ADM', 'TFL', 'GLU'};
type='s';
labelPrefix=fliplr([strcat('f',mOrder) strcat('s',mOrder)]); %To display
labelPrefixLong= strcat(labelPrefix,['_' type]); %Actual names
labels = cell(1,30*12);
for i = 1:length(labelPrefixLong)
    for j = 1:9
        labels{i*12+j-12}=sprintf('%s %d',labelPrefixLong{i},j);
    end
    for j = 10:12
        labels{i*12+j-12}=sprintf('%s%d',labelPrefixLong{i},j);
    end
end

group = groups{1};

%% Load EMG data for each interested epoch. 
baseData = nan(150,360,15); %stride x muscles x subject
adaData = nan(900,360,15);
postData = nan(400,360,15);
stepLengthAsym = nan(1450,1,15);
alphaSlow = nan(1450,1,15);
xSlow = nan(1450,1,15);

for sub = 1:15
    base = group.adaptData{sub}.getParamInCond(labels,{'TM base'});
    if (size(base,1) > 150)
        base = base(end-150:end-1,:); %last 150 except very last one?
    end
    baseData(:,:,sub) = base;
    ada = group.adaptData{sub}.getParamInCond(labels,{'Adaptation'});
    if (size(ada,1) > 900)
        ada = ada(2:901,:); % exclude the very first one, then take the first 900?
    end
    adaData(:,:,sub) = ada; %entire 900
    washout = group.adaptData{sub}.getParamInCond(labels,{'Washout'});    
    postData(:,:,sub) = washout(2:401,:); %first 400 except very first one?
    
    % Get step length asym
    sla = group.adaptData{sub}.getParamInCond({'netContributionNorm2'},{'TM base'});
    if (size(sla,1) > 150)
        sla = sla(end-150:end-1,:); %last 150 except very last one?
    end
    stepLengthAsym(1:150,:,sub) = zeros(150,1); % no learning in baseline.
    slaBase = median(sla(106:145,:,:),1); %last 41 except last 5

    sla = group.adaptData{sub}.getParamInCond({'netContributionNorm2'},{'Adaptation'});
    if (size(sla,1) > 900)
        sla = sla(2:901,:); %last 150 except very last one?
    end
    stepLengthAsym(151:1050,:,sub) = sla - slaBase;
    
    sla = group.adaptData{sub}.getParamInCond({'netContributionNorm2'},{'Washout'});
    stepLengthAsym(1051:1450,:,sub) = sla(2:401,:) - slaBase;
    
    % interporlate over the step length asymmetry to replace nans.
    t=1:size(stepLengthAsym,1); 
    nanidx=any(isnan(stepLengthAsym(:,:,sub)),2); %Any muscle missing
    stepLengthAsym(:,:,sub)=interp1(t(~nanidx),stepLengthAsym(~nanidx,:,sub),t,'linear',0); %Substitute nans
    newnan = find(any(isnan(stepLengthAsym(:,:,sub)),2));
    if (~isempty(newnan))
        fprintf("\nProblem: step length asymmetry\nSubject: %d\n",sub);
    end
    
    % get alpha positions (for slow leg only since only slow data available).
    alphaSBase = group.adaptData{sub}.getParamInCond({'alphaSlow'},{'TM slow'});
    alphaSBase = median(alphaSBase(end-44:end-5,:)); %last 40 except last 5.
    alphaSlow(1:150,:,sub) = zeros(150,1); %the baseline assumes no error signal
    alphaSAda = group.adaptData{sub}.getParamInCond({'alphaSlow'},{'Adaptation'});
    if (size(alphaSAda,1) > 900)
        alphaSAda = alphaSAda(2:901,:); %last 150 except very last one?
    end
    alphaSPost = group.adaptData{sub}.getParamInCond({'alphaSlow'},{'Washout'});
    alphaSlow(151:1050,:,sub) = alphaSAda - alphaSBase;
    alphaSlow(1051:1450,:,sub) = alphaSPost(2:401,:) - alphaSBase;
    
    % interporlate over the alpha slow o replace nans.
    t=1:size(alphaSlow,1); 
    nanidx=any(isnan(alphaSlow(:,:,sub)),2); %Any muscle missing
    alphaSlow(:,:,sub)=interp1(t(~nanidx),alphaSlow(~nanidx,:,sub),t,'linear',0); %Substitute nans
    newnan = find(any(isnan(alphaSlow(:,:,sub)),2));
    if (~isempty(newnan))
        fprintf("\nProblem alpha slow \nSubject: %d\n",sub);
    end
    
    % get the X positions (for slow, and fast washout since washout the leg swapped).
    xSBase = group.adaptData{sub}.getParamInCond({'XSlow'},{'TM slow'});
    xSBase = median(xSBase(end-44:end-5,:)); %last 40 except last 5.
    xSlow(1:150,:,sub) = zeros(150,1); %the baseline assumes no error signal
    xSAda = group.adaptData{sub}.getParamInCond({'XSlow'},{'Adaptation'});
    if (size(xSAda,1) > 900)
        xSAda = xSAda(2:901,:); %last 150 except very last one?
    end    
    xFPost = group.adaptData{sub}.getParamInCond({'XFast'},{'Washout'});
    xSlow(151:1050,:,sub) = xSAda - xSBase;
    xSlow(1051:1450,:,sub) = xFPost(2:401,:) - xSBase; %post is switched, slow is predictive of fast
    
    % interporlate over the Xslow to replace nans.
    t=1:size(xSlow,1); 
    nanidx=any(isnan(xSlow(:,:,sub)),2); %Any muscle missing
    xSlow(:,:,sub)=interp1(t(~nanidx),xSlow(~nanidx,:,sub),t,'linear',0); %Substitute nans
    newnan = find(any(isnan(xSlow(:,:,sub)),2));
    if (~isempty(newnan))
        fprintf("\nProblem X slow \nSubject: %d\n",sub);
    end   
end

clear xSBase xSAda xFPost alphaSBase alphaSAda alphaSPost sla slaBase

%% Interporlate over nan & compute the asymmetry component.
data={baseData, adaData, postData};
dataSym = cell(1,3);
for i = 1:3
    %Interpolate over NaNs %This is only needed if we want to run fast
    %estimations, or if we want to avoid all subjects' data at one
    %timepoint from being discarded because of a single subject's missing
    %data
    for j=1:15 %each subj
       t=1:size(data{i},1); 
       nanidx=any(isnan(data{i}(:,:,j)),2); %Any muscle missing
       data{i}(:,:,j)=interp1(t(~nanidx),data{i}(~nanidx,:,j),t,'linear',0); %Substitute nans
       newnan = find(any(isnan(data{i}(:,:,j)),2));
       if (~isempty(newnan))
            fprintf("\nProblem\nSubject: %d, Data: %d\n",j,i);
       end
    end

    %Compute asymmetry component
    aux=data{i}-fftshift(data{i},2); %slow- fast leg.
    dataSym{i}=aux(:,1:size(aux,2)/2,:); %strides x muscles x subjects
end
    
clear baseData adaData postData ada base washout data i sub aux j newnan nanidx t;

%% All data
Yf=[median(dataSym{1},3); median(dataSym{2},3);median(dataSym{3},3)]'; %(muscleLabels, 180) x strides
Uf=[zeros(size(dataSym{1},1),1);ones(size(dataSym{2},1),1);zeros(size(dataSym{3},1),1);]';%0 for B and P, 1 for A, 1 by strides

error = 4;
if error == 1
    errorEMGBase = median(dataSym{1}(end-44:end-5,:,:),1); %last 41 except last 5
    errorEMGBase = {dataSym{1} - errorEMGBase, dataSym{2} - errorEMGBase, dataSym{3} - errorEMGBase};
    Ue = [zeros(size(dataSym{1},1),1); median(errorEMGBase{2},3);median(errorEMGBase{3},3)]'; %no learning for baseline here
    Uf=[Uf;Ue]; %[Vasy, deltaEMG] by strides
elseif error == 2 %StepLengthAsym Base
    stepLengthAsymMedian = median(stepLengthAsym,3)';
    Uf = [Uf; stepLengthAsymMedian];
elseif error == 3 %TrailingLeg Base, alpha
    Uf = [Uf; median(alphaSlow,3)'];
elseif error == 4 %LeadingLeg Base, X
    Uf = [Uf; median(xSlow,3)'];
end

Yf=Yf(:,:); %Using only 400 of Post, use all created for now.
Uf=Uf(:,:);
% Just B and A
Y=Yf(:,1:1050);
U=Uf(:,1:1050);
clear errorEMGBase stepLengthAsymMedian
%% Section the dataout to different conditions (not sure for what?)

% %% Just P
% Yp=Yf(:,951:end);
% Up=Uf(:,951:end);
% %% P and some A
% Y_p=Yf(:,850:end);
% U_p=Uf(:,850:end);
% %% Median-filtered B, A
% binw=3;
% Y2=[medfilt1(median(dataSym{1},3),binw,'truncate'); medfilt1(median(dataSym{2},3),binw,'truncate')]';
%% Flat model:
clear model;
% [J,B,C,D,Q,R]=getFlatModel(Y,U);
[J,B,C,D,Q,R]=getFlatModel(Yf,Uf);
model{1}=autodeal(J,B,C,D,Q,R);
model{1}.name='Flat';
% Run the higher order models
for D1=1:5 %run up to order 5 for now
    fprintf("\n\n-----------Running model %d-----------\n",D1);
    % Identify
    tic
    opts.robustFlag=false;
    opts.Niter=1500;
    opts.outlierReject=false;
    opts.fastFlag=true;
    %[fAh,fBh,fCh,D,fQh,R,fXh,fPh]=randomStartEM(Yf,Uf,D1,10,opts); %Slow/true EM
    [fAh,fBh,fCh,D,fQh,R,fXh,fPh]=randomStartEM(Yf,Uf,D1,opts); %Slow/true EM
    logL=dataLogLikelihood(Y,U,fAh,fBh,fCh,D,fQh,R,fXh(:,1),fPh(:,:,1));
    model{D1+1}.runtime=toc;
    %[J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
    [J,B,C,X,~,Q,P] = canonize(fAh,fBh,fCh,fXh,fQh,fPh);
    model{D1+1}=autodeal(J,B,C,D,X,Q,R,P,logL);
    model{D1+1}.name=['EM (iterated,all,' num2str(D1) ')']; %Robust mode does not do fast filtering
end
%%
% save EMrealDimCompare1500mean.mat
%% COmpare
vizModels(model(1:5))
%%
%vizDataFit(model([4:-1:1]),Y,U)
%vizDataFit(model([4:-1:1]),Y_p,U_p)
vizDataFit(model(:),Yf,Uf)

%TODO: whats the EMG usued? 