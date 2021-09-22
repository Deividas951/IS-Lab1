% Classification using perceptron

% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Calculate for each image, colour and roundness
% For Apples
% 1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
% 2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
% 3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
% 4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
% 5th apple image(A5)hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
% 6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
% 7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
% 8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
% 9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];
% estimated features are stored in matrix P:
P=[x1;x2];
xt1=[hsv_value_A4 hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9 hsv_value_P3 hsv_value_P4];
xt2=[metric_A4 metric_A6 metric_A7 metric_A8 metric_A9 metric_P3 metric_P4];

%Desired output vector
T=[1;1;1;-1;-1]; % <- ČIA ANKSČIAU BUVO KLAIDA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%% train single perceptron with two inputs and one output

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

% calculate weighted sum with randomly generated parameters
v1 = x1(1)*w1 + x2(1)*w2 + b; %v1 = <...>; % write your code here
% calculate current output of the perceptron 
if v1 > 0
	y1 = 1;
else
	y1 = -1;
end
% calculate the error
e1 = T(1) - y1;

% repeat the same for the rest 4 inputs x1 and x2
% calculate wieghted sum with randomly generated parameters
v2 = x1(2)*w1 + x2(2)*w2 + b % v2 = <...> ; % write your code here
% calculate current output of the perceptron 
if v2 > 0
	y2 = 1;
else
	y2 = -1;
end
% calculate the error
e2 = T(2) - y2;
%--------------------------------------------------------------------------
v3 = x1(3)*w1 + x2(3)*w2 + b % <...> write the code for another 3 inputs
if v3 > 0
	y3 = 1;
else
	y3 = -1;
end
e3 = T(3) - y3;
%--------------------------------------------------------------------------
v4 = x1(4)*w1 + x2(4)*w2 + b % <...> write the code for another 3 inputs
if v4 > 0
	y4 = 1;
else
	y4 = -1;
end
e4 = T(4) - y4;
%--------------------------------------------------------------------------
v5 = x1(5)*w1 + x2(5)*w2 + b % <...> write the code for another 3 inputs
if v5 > 0
	y5 = 1;
else
	y5 = -1;
end
e5 = T(5) - y5;
%--------------------------------------------------------------------------
% calculate the total error for these 5 inputs 
e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5);
g = 0.1;
cycleCount = 1;
% write training algorithm
while e ~= 0 % executes while the total error is not 0
	% here should be your code of parameter update
%   calculate output for current example
% 
%   calculate error for current example
% 
%   update parameters using current inputs ant current error
w1 = w1 + g*e1*x1(1);   % 	w1 = 
w2 = w2 + g*e1*x2(1);   %   w2 = 
b  = b  + g*e1*1;         %   b = 
%--------------------------------------------------------------------------
w1 = w1 + g*e2*x1(2);   
w2 = w2 + g*e2*x2(2);   
b  = b  + g*e2*1;    
%--------------------------------------------------------------------------
w1 = w1 + g*e3*x1(3);   
w2 = w2 + g*e3*x2(3);   
b  = b  + g*e3*1;   
%--------------------------------------------------------------------------
w1 = w1 + g*e4*x1(4);   
w2 = w2 + g*e4*x2(4);   
b  = b  + g*e4*1;   
%--------------------------------------------------------------------------
w1 = w1 + g*e5*x1(5);   
w2 = w2 + g*e5*x2(5);   
b  = b  + g*e5*1;   
% 
%   Test how good are updated parameters (weights) on all examples used for training
%   calculate outputs and errors for all 5 examples using current values of the parameter set {w1, w2, b}
%   calculate 'v1', 'v2', 'v3',... 'v5'
% 
%   calculate 'y1', ..., 'y5'
%     
%   calculate 'e1', ... 'e5'

v1 = x1(1)*w1 + x2(1)*w2 + b;
if v1 > 0
	y1 = 1;
else
	y1 = -1;
end
e1 = T(1) - y1;
%--------------------------------------------------------------------------
v2 = x1(2)*w1 + x2(2)*w2 + b;
if v2 > 0
	y2 = 1;
else
	y2 = -1;
end
e2 = T(2) - y2;
%--------------------------------------------------------------------------
v3 = x1(3)*w1 + x2(3)*w2 + b;
if v3 > 0
	y3 = 1;
else
	y3 = -1;
end
e3 = T(3) - y3;
%--------------------------------------------------------------------------
v4 = x1(4)*w1 + x2(4)*w2 + b;
if v4 > 0
	y4 = 1;
else
	y4 = -1;
end
e4 = T(4) - y4;
%--------------------------------------------------------------------------
v5 = x1(5)*w1 + x2(5)*w2 + b;
if v5 > 0
	y5 = 1;
else
	y5 = -1;
end
e5 = T(5) - y5;
%--------------------------------------------------------------------------   
	% calculate the total error for these 5 inputs 
	e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5);
    cycleCount = cycleCount + 1;
end

%% Testing----------------------------------------------------------------------
% Testing data
%xt1 = [hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9 hsv_value_P3 hsv_value_P4];
%xt2 = [metric_A4 metric_A5 metric_A6 metric_A7 metric_A8 metric_A9 metric_P3 metric_P4];



vt1 = xt1(1)*w1 + xt2(1)*w2 + b;
if vt1 > 0
	yt1 = 1;
else
	yt1 = -1;
end
et1 = T(1) - yt1;
%--------------------------------------------------------------------------
vt2 = xt1(2)*w1 + xt2(2)*w2 + b;
if vt2 > 0
	yt2 = 1;
else
	yt2 = -1;
end
et2 = T(1) - yt2;
%--------------------------------------------------------------------------
vt3 = xt1(3)*w1 + xt2(3)*w2 + b;
if vt3 > 0
	yt3 = 1;
else
	yt3 = -1;
end
et3 = T(1) - yt3;
%--------------------------------------------------------------------------
vt4 = xt1(4)*w1 + xt2(4)*w2 + b;
if vt4 > 0
	yt4 = 1;
else
	yt4 = -1;
end
et4 = T(1) - yt4;
%--------------------------------------------------------------------------
vt5 = xt1(5)*w1 + xt2(5)*w2 + b;
if vt5 > 0
	yt5 = 1;
else
	yt5 = -1;
end
et5 = T(1) - yt5;
%--------------------------------------------------------------------------
vt6 = xt1(6)*w1 + xt2(6)*w2 + b;
if vt6 > 0
	yt6 = 1;
else
	yt6 = -1;
end
et6 = T(1) - yt6;
%--------------------------------------------------------------------------
vt7 = xt1(7)*w1 + xt2(7)*w2 + b;
if vt7 > 0
	yt7 = 1;
else
	yt7 = -1;
end
et7 = T(1) - yt7;


%% Naive Bayes classificator.
%COLOURS
%1 - RedYellow
%2 - Yellow
%3 - DarkRed/Yellow
%4 - Green
%SIZES
%1 - OvalShape
%2 - PearShape

x1=[1 2 3 2 4]; %Colour vector [1 2 3 2 4]
x2=[1 1 1 2 2]; %Size vector [1 1 1 2 2]
%x1=[{'Yellow' 'Yellow' 'DarkRedYellow' 'Yellow' 'Green'}]; %Colour vector
%x2=[{'OvalShape' 'OvalShape' 'OvalShape' 'PearShape' 'PearShape'}]; %Size vector
%T = [{'Apple' 'Apple' 'Apple' 'PearShape' 'PearShape'}];
Target=[1,1,1,0,0]; %Target - 1:Apple, 0:Pear
%Table = [x1; x2; Target]';
AppleCount = 3; %Apple count in training data
PearCount = 2; %Pear count in training data

%x1x2 Apple
x1a1 = 0;
for ind = 1:5
    if Target(ind) == 1 && x1(ind) == x1(1)
        x1a1 = x1a1+1;
    end
end
x2a1 = 0;
for ind = 1:5
    if Target(ind) == 1 && x2(ind) == x2(1)
        x2a1 = x2a1+1;
    end
end
%x1x2 Pear
x1p1 = 0;
for ind = 1:5
    if Target(ind) == 0 && x1(ind) == x1(1)
        x1p1 = x1p1+1;
    end
end
x2p1 = 0;
for ind = 1:5
    if Target(ind) == 0 && x2(ind) == x2(1)
        x2p1 = x2p1+1;
    end
end

v1Apple = (x1a1/AppleCount)*(x2a1/AppleCount);
v1Pear = (x1p1/PearCount)*(x2p1/PearCount);

if v1Apple > v1Pear
    y1 = 1
else
    y1 = 0
end

%--------------------------------------------------------------------------
%x1x2 Apple
x1a2 = 0;
for ind = 1:5
    if Target(ind) == 1 && x1(ind) == x1(2)
        x1a2 = x1a2+1;
    end
end
x2a2 = 0;
for ind = 1:5
    if Target(ind) == 1 && x2(ind) == x2(2)
        x2a2 = x2a2+1;
    end
end
%x1x2 Pear
x1p2 = 0;
for ind = 1:5
    if Target(ind) == 0 && x1(ind) == x1(2)
        x1p2 = x1p2+1;
    end
end
x2p2 = 0;
for ind = 1:5
    if Target(ind) == 0 && x2(ind) == x2(2)
        x2p2 = x2p2+1;
    end
end

%--------------------------------------------------------------------------


v2Apple = (x1a2/AppleCount)*(x2a2/AppleCount);
v2Pear = (x1p2/PearCount)*(x2p2/PearCount);

if v2Apple > v2Pear
    y2 = 1
else
    y2 = 0
end

%--------------------------------------------------------------------------
%x1x2 Apple
x1a3 = 0;
for ind = 1:5
    if Target(ind) == 1 && x1(ind) == x1(3)
        x1a3 = x1a3+1;
    end
end
x2a3 = 0;
for ind = 1:5
    if Target(ind) == 1 && x2(ind) == x2(3)
        x2a3 = x2a3+1;
    end
end
%x1x2 Pear
x1p3 = 0;
for ind = 1:5
    if Target(ind) == 0 && x1(ind) == x1(3)
        x1p3 = x1p3+1;
    end
end
x2p3 = 0;
for ind = 1:5
    if Target(ind) == 0 && x2(ind) == x2(3)
        x2p3 = x2p3+1;
    end
end
v3Apple = (x1a3/AppleCount)*(x2a3/AppleCount);
v3Pear = (x1p3/PearCount)*(x2p3/PearCount);

if v3Apple > v3Pear
    y3 = 1
else
    y3 = 0
end
%--------------------------------------------------------------------------
%x1x2 Apple
x1a4 = 0;
for ind = 1:5
    if Target(ind) == 1 && x1(ind) == x1(4)
        x1a4 = x1a4+1;
    end
end
x2a4 = 0;
for ind = 1:5
    if Target(ind) == 1 && x2(ind) == x2(4)
        x2a4 = x2a4+1;
    end
end
%x1x2 Pear
x1p4 = 0;
for ind = 1:5
    if Target(ind) == 0 && x1(ind) == x1(4)
        x1p4 = x1p4+1;
    end
end
x2p4 = 0;
for ind = 1:5
    if Target(ind) == 0 && x2(ind) == x2(4)
        x2p4 = x2p4+1;
    end
end


v4Apple = (x1a4/AppleCount)*(x2a4/AppleCount);
v4Pear = (x1p4/PearCount)*(x2p4/PearCount);

if v4Apple > v4Pear
    y4 = 1
else
    y4 = 0
end
%--------------------------------------------------------------------------
%x1x2 Apple
x1a5 = 0;
for ind = 1:5
    if Target(ind) == 1 && x1(ind) == x1(5)
        x1a5 = x1a5+1;
    end
end
x2a5 = 0;
for ind = 1:5
    if Target(ind) == 1 && x2(ind) == x2(5)
        x2a5 = x2a5+1;
    end
end
%x1x2 Pear
x1p5 = 0;
for ind = 1:5
    if Target(ind) == 0 && x1(ind) == x1(5)
        x1p5 = x1p5+1;
    end
end
x2p5 = 0;
for ind = 1:5
    if Target(ind) == 0 && x2(ind) == x2(5)
        x2p5 = x2p5+1;
    end
end

v5Apple = (x1a5/AppleCount)*(x2a5/AppleCount);
v5Pear = (x1p5/PearCount)*(x2p5/PearCount);

if v5Apple > v5Pear
    y5 = 1
else
    y5 = 0
end



