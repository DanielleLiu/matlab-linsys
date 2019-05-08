function [fh] = vizCVDataLikelihood(model,testSet,method)
if ~iscell(testSet)
    testSet={testSet};
end
if ~isa(model{1},'struct') %Has to be cell
  %To do: convert struct to model
end
if length(testSet)~=size(model,2)
    error('Number of models and testSets is not the same')
end
M=max(cellfun(@(x) size(x.A,1),model(:)));
fh=figure('Units','Normalized','OuterPosition',[.25 .4 .5 .3]);

%LogL, BIC, AIC
mdl=model;
Md=length(testSet);
if nargin<3 || isempty(method)
method='logL';
end
for kd=1:Md %One row of subplots per dataset
    dFit=cellfun(@(x) x.fit(testSet{kd}),model(:,kd),'UniformOutput',false); %Fit with improper initial condition
    switch method
        case 'logL'
            logLtest=cellfun(@(x) x.logL,dFit);
        case 'oneAheadRMSE'
            logLtest=cellfun(@(x) sqrt(mean(sum(x.oneAheadResidual.^2))),dFit);
        case 'detRMSE'
            logLtest=cellfun(@(x) sqrt(mean(sum(x.deterministicResidual.^2))),dFit);
    end
    yy=logLtest;
    yy=yy-min(yy);
    nn=['\Delta ' method];
    subplot(Md,1,kd)
    hold on
    Mm=length(mdl);
    DeltaIC=yy-max(yy);
    modelL=exp(DeltaIC);
    w=modelL/sum(modelL); %Computes bayes factors, or Akaike weights. See Wagenmakers and Farrell 2004, Akaike 1978, Kass and Raferty 1995
    for k=1:Mm
        set(gca,'ColorOrderIndex',k)
        bar2=bar([k*100],yy(k),'EdgeColor','none','BarWidth',100);
        text((k)*100,.05*(yy(k)),[num2str(yy(k),6)],'Color','w','FontSize',8,'Rotation',90)
    end
    set(gca,'XTick',[1:Mm]*100,'XTickLabel',cellfun(@(x) x.name, mdl,'UniformOutput',false),'XTickLabelRotation',90)
    title([nn])
    grid on
    axis tight;
    aa=axis;
    axis([aa(1:2) 0 1.1*max(yy)])
end
end
