#include <iostream>
#include "lwesamples.h"
#include "tfhe.h"
#include "tlwe.h"
#include <vector>
#include <fstream>
#include <cmath>
#include <sstream>
#include "math.h"
#include <ctime>

#define input_begin 25
#define input_rows 125
#define target_begin 0
#define target_rows 5
using namespace std;

const int input_layer = 100;  //输入单元个数
const int output_layer = 5;   //输出单元个数
const int train_examples = 1504;
const int test_examples = 2504;  //测试样本量
#define alpha pow(2. , -20)

void readinginput(vector<vector<int>> &input, ifstream &ifile)
{
	string str;
	int rows = 0;
	while (getline(ifile, str) && rows<input_rows)
	{
		vector<int> temp;
		istringstream bbb(str);
		int aaa;
		while (bbb >> aaa)
		{
			temp.push_back(aaa);
		}
		input.push_back(temp);
		++rows;
	}
}

//round函数
double round(double r)

{
	double temp;
	temp = (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
	if(temp > 2)
		return 2;
	else if(temp < 0)
		return 0;
	else
		return temp;
}
//void cal_sum(vector<vector<double>> &W , vector<int> &input , vector<double> &output);

int main()
{
    cout << "linear regression is coming !" << endl;
    clock_t starttime, endtime;

    const int mini_lamba = 128;
    const int spc_msg = 700;
    TFheGateBootstrappingParameterSet *params = new_default_gate_bootstrapping_parameters(mini_lamba);
    const LweParams *in_out_params = params->in_out_params;

    TFheGateBootstrappingSecretKeySet *secret = new_random_gate_bootstrapping_secret_keyset(params);

        //export the secret key to file for later use
    FILE* secret_key = fopen("secret.key","wb");
    export_tfheGateBootstrappingSecretKeySet_toFile(secret_key, secret);
    fclose(secret_key);


    //读取权重
	vector<vector<double>> W;
	ifstream weight_file("weight100-1.txt");
	if(!weight_file)   //打开文件失败 则返回-1
	{
		cout<<"oops,fail to open the lable file!\n";
		return -1;
	}
	string str1;  //储存一行数据
	while(getline(weight_file , str1))
	{
		istringstream input(str1);
		vector<double> temp;
		double a;
		while(input>>a)
			temp.push_back(a*100);
		W.push_back(temp);
	}
	weight_file.close();

 	//读取输入和输出
	ifstream ifile("new2.txt");
	vector<vector<int>> input;
	readinginput(input, ifile);
	ifile.close();

	int input_count;
	input_count = input.size();
	cout<<"input_size is "<<input_count<<endl;

	//读取真实标签
	ifstream tfile("saved_target_all.txt");
	if(!tfile)
	{
		cout<<"saved_target_all.txt open failed~"<<endl;
		return -1;
	}
	vector<vector<int>> Y;
	string sstr;
	while(getline(tfile , sstr))
	{
		istringstream input(sstr);
		vector<int> temp;
		int b;
		int row=0;
		while(input >> b)
		{
			if(row < target_begin)
				row++;
			else if(target_begin <= row  && row<target_rows)
			{
				temp.push_back(b);
				row++;
			}
			else
				break;
		}
		Y.push_back(temp);
	}
	tfile.close();
	int output_size;
	output_size = Y.size();
	cout<<"output_size is "<<output_size<<endl;

    int count=0; //统计预测正确数
    Torus32 mu_tmp;
    mu_tmp = modSwitchToTorus32(1 , spc_msg);
    starttime = clock();
    clock_t each_stattime , each_endtime;
    double total_enctime = 0.0;
    double total_cau = 0.0;
    double total_dec = 0.0;
    for(int test_num=0; test_num<1004; test_num++)
		{
		    each_stattime =clock();
		    vector<int> input_cur(input_layer); //当前数据输入
			int it = input_begin;
			while (it<input_rows)
			{
				input_cur[it - input_begin] = input[it][test_num];
				++it;
			}

            LweSample *enc_input, *mul_sum;
            enc_input = new_LweSample_array(input_layer , in_out_params);
            mul_sum = new_LweSample_array(output_layer , in_out_params);
            //int plain_input , plain_W;
            clock_t startenc, endenc;
            startenc = clock();
            for(int i = 0; i < input_layer; i++)
            {
            Torus32 mu;
            int plain_input = input_cur[i] ;
            mu = modSwitchToTorus32(plain_input, spc_msg);
            lweSymEncrypt(enc_input + i , mu , alpha , secret->lwe_key);
            }
            endenc = clock();
            total_enctime += (double)(endenc - startenc) /CLOCKS_PER_SEC;

            clock_t startcal, enccal;
            startcal = clock();
            for(int i =0; i<input_layer; i++)
            {
                for(int j=0; j<output_layer; j++)
                {
                     int plain_W = W[i][j];
                     lweAddMulTo(mul_sum + j , plain_W , enc_input + i , in_out_params);
                }

            }
            enccal = clock();
            total_cau += (double)(enccal - startcal) / CLOCKS_PER_SEC;

        Torus32 dec_output;
        double lhl;
        clock_t startdec, enddec;
        startdec = clock();
        for(int j=0; j < output_layer; j++)
        {

         dec_output = lwePhase(mul_sum + j, secret->lwe_key);
         //cout<<"dec_output "<<dec_output << endl;
         //cout<<"  mu_tmp    " << mu_tmp <<endl;
         lhl = (float) dec_output / mu_tmp;
         //cout<<"decrypted mul_sum is "<<lhl<<endl;

         int p = round((float)lhl / 100);
         int k = Y[test_num][j];   //标签值
         //cout<<"the prediction is "<<p<<", the truth is " <<k<<endl;
         if(p == k)
            count++;
         // cout<<"---------------------------------"<<endl;
        }
        enddec = clock();
        total_dec += (double)(enddec - startdec)/CLOCKS_PER_SEC;
        each_endtime = clock();
        //cout<< "each prediction time is "<<(double)(each_endtime - each_stattime) / CLOCKS_PER_SEC<<" s"<<endl;
		}
		endtime = clock();
		cout<<"total time is "<<(double)(endtime - starttime) /CLOCKS_PER_SEC<<" s"<<endl;

		cout<<"total enc time is "<<total_enctime<<endl;
		cout<<"total cal time is "<<total_cau<<endl;
		cout<<"total dec time is "<<total_dec<<endl;

    double acc =0.0;
	acc = (float)count / (1004 * output_layer);
	cout<<"the accuracy is "<<acc<<endl;
	//cout<<"the max_output is  "<<max_output<<endl;
	system("pause");

    return 0;
}
