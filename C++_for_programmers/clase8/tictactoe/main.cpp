#include <iostream>
#include "tictactoe.h"
#include "mainFunctions.h"

using namespace std;

int main()
{
	string oUser = promptUser('o');
    string xUser = promptUser('x');
    
    Tictactoe tictactoe;

    tictactoe.setOUser(oUser);
    tictactoe.setXUser(xUser);

    tictactoe.printInfo();

    while(tictactoe.isGameOver() == false)
    {
        int row, column;

        getSymbolPosition(tictactoe.getOUser(),'o', row, column);
        tictactoe.setGameSpace(row - 1, column - 1, 'o'); // zero based matrix
        tictactoe.printInfo();

        if(tictactoe.isGameOver()) break;

        getSymbolPosition(tictactoe.getXUser(), 'x', row, column);
        tictactoe.setGameSpace(row - 1, column - 1, 'x'); // zero based matrix
        tictactoe.printInfo();
        
    }
    // get and print winner
    cout << tictactoe.getWinner() << endl;
     
	return 0;
}
