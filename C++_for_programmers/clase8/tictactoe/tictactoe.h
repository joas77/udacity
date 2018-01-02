#include <iostream>
#include <iomanip> 
using namespace std;

const int ROWS = 4;
const int COLUMNS = 4;

class Tictactoe
{
    char gameSpace[ROWS][COLUMNS];
    string mXUser;
    string mOUser;
    string mWinner;
    int fourInRow(char value); //four 'x' or 'o's in any row 'wins'
    int fourInDiag(char value);
    bool isFilled();
public:
    Tictactoe(); //initialize the board with '-' in all 16 spaces
    void setGameSpace(int row,int column, char value); //x,y,or '-' in each game square
    char getGameSpace(int row,int column);
    void printInfo(); //print the game board in a 4x4 matrix
    void setXUser(string user);
    void setOUser(string user);

    string getXUser();
    string getOUser();
    string getWinner();

    bool isGameOver();
};

// constructor
Tictactoe::Tictactoe()
{
    mXUser = "x user";
    mOUser = "o user";

    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            gameSpace[i][j] = '-';
}

string Tictactoe::getWinner()
{
    return mWinner;
}

bool Tictactoe::isGameOver()
{
    bool retValue = false;

    if(fourInRow('x')||fourInDiag('x'))
    {
        //TODO: set x player as winner
        mWinner = "congratulations " + mXUser + " you are the winner!";
        retValue = true;
    }
    else if(fourInRow('o')||fourInDiag('o'))
    {
        mWinner = "congratulations " + mOUser + " you are the winner!";
        retValue = true;
    }
    else if(isFilled())
    {
        mWinner = "is tied! nobody wins";
        retValue = true;
    }

    return retValue;
}

bool Tictactoe::isFilled()
{
    for(int i=0; i < ROWS; i++)
    {
        for(int j=0; j < COLUMNS; j++)
        {
            if(gameSpace[i][j] == '-') 
            {
                //cout << gameSpace[i][j];
                return false;
            }
        }
    }
    return true;
}

void Tictactoe::setOUser(string user)
{
    mOUser = user;
}

void Tictactoe::setXUser(string user)
{
    mXUser = user;
}

string Tictactoe::getOUser()
{
    return mOUser;
}

string Tictactoe::getXUser()
{
    return mXUser;
}

void Tictactoe::setGameSpace(int row, int column, char value)
{
    gameSpace[row][column] = value;
}

char Tictactoe::getGameSpace(int row,int column)
{
    return gameSpace[row][column];
}

int Tictactoe::fourInRow(char value)
{
    for(int i=0; i<4; i++)
    {
        int xcounter = 0;
        for(int j=0; j<4; j++)
        {
            if( value == gameSpace[i][j] ) xcounter++;
        }
        if(xcounter == 4) return 1;
        cout<<endl;
    }    
    return 0;
}

int Tictactoe::fourInDiag(char value)
{
    int leftToRigthDiagCount=0;
    int rightToLeftDiagCount=0;
    int retValue = 0;
    
    for(int i = 0; i < 4; i++)
    {
        if(gameSpace[i][i] == value) leftToRigthDiagCount++;
        
        if(gameSpace[i][4-(i+1)] == value) rightToLeftDiagCount++;
    }

    if(leftToRigthDiagCount == 4 || rightToLeftDiagCount == 4)
    {
        retValue = 1;
    }

    return retValue;
}

void Tictactoe::printInfo()
{
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
        {
            cout<<gameSpace[i][j] << " ";
        }
        cout<<endl;
    }
}
