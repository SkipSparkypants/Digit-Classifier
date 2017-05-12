clc, clear
load USPS.mat
%initialize matrices
train_aves(1:256,1:10) = 0;
test_classif(1:10, 1:4649) = 0;
test_classif_res(1, 1:4649) = 0;
test_confusion(1:10,1:10) = 0;
test_svd17_confusion(1:10,1:10) = 0;
test_svd17res(1:10, 1:4649) = 0;
%part 1
%display first 16 images of test_patterns
for k = 1:16
temp = reshape(test_patterns(:,k),[16,16]);
figure(1)
subplot(4,4,k)
imagesc(temp')
end
%part 2
%training step
for k = 1:10    %for each digit
    temp2 = train_patterns(:, train_labels(k,:)==1); %pool image k into temp2
    temp3 = mean(temp2');   %average image k
    for(i = 1:256)
        train_aves(i,k) = temp3(1,i);   %add average values for each pixel i into col k
    end
    figure(2)
    temp4 = reshape(train_aves(:,k),[16,16]);   %plot average images 
    subplot(2,5,k)
    imagesc(temp4')
    test_classif(k,:) = sum((test_patterns-repmat(train_aves(:,k),[1 4649])).^2); %fill test_classif w/ euclidean distance between test_patterns, train_aves
end
%part 3
for j = 1:4649  % for each test image
     %compute classification results
     [tmp, ind] = min(test_classif(:,j)); %find min row j
     test_classif_res(1,j) = ind;   %store result in test_classif_res
end
for k = 1:10    % for each digit
    %compute test_confusion matrix
    tmp = test_classif_res(test_labels(k,:)==1);
    [m,n] = size(tmp);
    for i = 1:n     %for each image of digit k
        if(tmp(1,i) == k)   %if tmp[i] = k , increment test_confusion[k,k]
            test_confusion(k,k) = test_confusion(k,k) + 1;
        else    %if tmp[i] != k, increment test_confusion[k,tmp(1,i)]
            test_confusion(k,tmp(1,i)) = test_confusion(k,tmp(1,i)) + 1;
        end
    end
end
%part 4
%training step
for k=1:10  %for each digit
    %compute svd 
    [train_u(:,:,k),tmp,tmp2] = svds(train_patterns(:,train_labels(k,:)==1),17);
    test_svd17(:,:,k) = transpose(train_u(:,:,k)) * test_patterns;
    tmp = train_u(:,:,k)*test_svd17(:,:,k); %r17 approximation
    for j = 1:4649
        %compute 2-norm of residual error, store in test_svd17_res
        Ej = norm(test_patterns(:,j) - tmp(:,j)); %error for digit j
        test_svd17res(k, j) = Ej;
    end
end
%compute test_svd17_confusion matrix``  
for i = 1:4649  %for each test image
    digit = test_labels(:,i) == 1;  %find actual value of image, store in digit
    [m,n] = min(test_svd17res(:,i)); %find estimated digit, store in n
    test_svd17_confusion(digit,n) = test_svd17_confusion(digit,n) + 1;
end


