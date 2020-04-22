function [fh,ResAll] = vizDataFitAcrossModels(model,Y,U)
  if ~iscell(model)
    model={model};
  end
  if isa(model{1},'struct') %For back-compatibility
    model=cellfun(@(x) linsys.struct2linsys(x),model,'UniformOutput',false);
  end

    fh=figure('Units','Normalized','OuterPosition',[0 0 1 1]);

    Nm=length(model); %Number of models
    setdataSets = cell(1,Nm);
    for i=1:Nm
    %'KF' %Kalman filter
    %KS Kalman smoother
    %KP Kalman one-ahead predictor
    %For this case, seems to be the same?
      datSet = dset(U{i},Y{i});
      setdataSets{i} = datSet;
%       dFit{i}=model{i}.fit(datSet,[],'KP'); %oneAhead
      dSmooth{i}=model{i}.fit(datSet,[],'KS'); %oneAhead
    end

%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
mid=ones(1,3);
N=100;
map=[ex1.*[N:-1:1]'/N + mid.*[0:N-1]'/N; mid; ex2.*[0:N-1]'/N + mid.*[N:-1:1]'/N];

%% Measures of output error:
%dataVariance=norm(datSet.out,'fro');
datSet = setdataSets{1};
rr=datSet.flatResiduals; %construct a flat model and then compute the residuals
residualReference=sqrt(sum(nansum(rr.^2))); %Residuals from flat model
dY=diff(datSet.out,[],2);
dU=diff(datSet.in,[],2); %in 2xd, datasetU1-U0
idx=find(dU(1,:)==0); %logical 1 (true) if dU is all 0 (all us are the same)
N=length(idx);
M=length(~idx);
dY=dY(:,idx); %most of the time du not all 0, will index no columns, return []
baseResiduals=sqrt(.5)*norm(dY,'fro')*(N+M+1)/N; %Residuals from one-ahead data, its coavariance is a lower bound of R+.5*CQC'
%resDet=cellfun(@(x) norm(x.simulate(datSet.in,[],true).out-datSet.out,'fro'),model)/dataVariance;
%5 (leng of models) x 2 (2 plots each model)
Nx=2; Ny=1;
ResAll = cell(1,Nm);
for ll=1%:2
    for k=1:Nm
        switch ll
            case 1 %Deterministic output RMSE
                iC=dSmooth{k}.stateEstim.getSample(1); %MLE estimate of init cond
                %iC=[]; %Start from 0?
                U = setdataSets{k}.in;
                [simSet]=model{k}.simulate(U,iC,true,true);
                res=datSet.out -simSet.out;
                rmseTimeCourse=nansum(res.^2);
                aux1=sqrt(sum(nansum(res.^2)))/residualReference;
                ResAll{k} = res;
                %aux1=sum(sum(abs(res)));
                tt={'Deterministic output error (RMSE, mov. avg.)'};
            case 3 % MLE state output error
                tt=('KS one-ahead output error (RMSE, mov. avg.)');
            case 2 %One ahead error
                [modelOut]=dFit{k}.output;
                res=modelOut-datSet.out;
                rmseTimeCourse=nansum(res.^2);
                aux1=sqrt(sum(nansum(res.^2)))/residualReference;
                %aux1=sum(sum(abs(res)));
                tt={'KF prediction error (RMSE, mov. avg.)'};
        end
%         subplot(Nx,Ny,2+(2*ll-2)*Ny) %Time-course of residuals
        subplot(Nx,Ny,ll) 
        hold on
        set(gca,'ColorOrderIndex',k)
%         s1=scatter(1:length(rmseTimeCourse),sqrt(rmseTimeCourse),5,'filled','MarkerEdgeColor','none','MarkerFaceAlpha',.5);
        p1 = plot(sqrt(rmseTimeCourse));%,'MarkerFaceColor',s1.MarkerFaceColor);
        if k==1
            title(tt)
            axis tight
            grid on
%             set(gca,'YScale','log')
        elseif k ==length(model)
            legendString = cell(1,k);
            for i=1:k
                legendString{i} = sprintf("Model %d",i);
            end
            legend(legendString);
        end
        subplot(Nx,Ny,2) %Bars of residuals
        hold on;
        set(gca,'ColorOrderIndex',1);
        RMSE=aux1; %Normalized Frobenius norm of residual as % of data variance
        bar2=bar(k,RMSE,'EdgeColor','k','BarWidth',1,'FaceColor',p1.Color);
        %text(k,.9*RMSE,[num2str(RMSE,4)],'Color','w','FontSize',6,'Rotation',270)
        set(gca,'XTick',1:length(model),'XTickLabel',cellfun(@(x) x.name,model,'UniformOutput',false))
        if k==length(model)
            axis tight
            grid on
            xlabel("Models");
            ylabel("Normalized Residual as % of Data Variance")
            tickLabels = cell(1,k);
            for i=1:k
                tickLabels{i} = sprintf("%d",i-1);
            end
            set(gca,'XTick',1:length(model),'XTickLabel',tickLabels);
%             set(gca,'YScale','log')
        end
    end
    subplot(Nx,Ny,Ny+ll)%subplot(Nx,Ny,2+(ll-1)*2*Ny) %Bars of residuals
%     hold on
%     plot([1,length(model)],[1 1]*baseResiduals/residualReference,'k--')
end
end
