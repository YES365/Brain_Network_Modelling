function [t,p,DF,beta,ci1,ci2] = LMMFit(tb,formula_input,x_name)
lmm = fitlme(tb,formula_input);
lmm_Coeff = lmm.Coefficients(contains(lmm.Coefficients.Name,x_name),:);
t = [lmm_Coeff.tStat];
p= [lmm_Coeff.pValue];
DF = [lmm_Coeff.DF];
beta = [lmm_Coeff.Estimate];
ci1= [lmm_Coeff.Lower];
ci2= [lmm_Coeff.Upper];
end

