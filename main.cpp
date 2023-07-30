#include"util/dataset.h"
#include"util/neuron.h"
#include"util/net.h"
#include <iostream>


void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}

int main()
{
	Dataset trainData("Data.txt");
	vector<unsigned> netShape;

    netShape.push_back(2);
    netShape.push_back(4);
    netShape.push_back(1);
	Net myNet(netShape);

    cout<<endl<<"====Neural network shape===="<<endl;
    for(unsigned n: netShape)
    {
        for(unsigned i = 0; i < n; ++i) cout<<"o ";
        cout<<endl;
    }
    cout<<"============================"<<endl;

	vector<double> input, label, output;
	int iteration = 0;
	while(!trainData.isEof())
	{
		++iteration;

		if(trainData.getNextInputs(input) != netShape[0])
			break;

		myNet.feedForward(input);
		myNet.getResults(output);
		trainData.gettarget(label);

		assert(label.size() == netShape.back());
		myNet.backProp(label);

        if(iteration % 100 == 0)
        {
            cout << endl << "Iteration" << iteration;
            showVectorVals(": Inputs :", input);
            showVectorVals("Outputs:", output);
            showVectorVals("Targets:", label);
            cout << "Recent average error: " << myNet.getError() << endl;
        }
	}

	cout << endl << "Training done" << endl;

}
