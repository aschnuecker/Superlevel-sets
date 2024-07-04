function[output, cdf_mix] = mixQ(q,z,w,mu,Sigma)

M = size(w,1);
cdf_mix = zeros(M,1);

 % normal distributions
for m = 1:M
    cdf_mix(m) = w(m)*normcdf(z,mu(m),sqrt(Sigma(m)));
end
output = sum(cdf_mix) - q;





