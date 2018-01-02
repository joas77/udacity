#include <iostream>

std::string promptUser(char symbol);
void getSymbolPosition(std::string user, char symbol, int& row, int& column);

std::string promptUser(char symbol)
{
    std::string user;
    std::cout << "name of "<< symbol << "'s user: ";
    std::cin >> user;
    std::cout << "Welcome " << user <<"!" << std::endl;
    return user;
}

void getSymbolPosition(std::string user, char symbol, int& row, int& column)
{
    std::cout << user << " where do you like to place your " << symbol << " ?"<< std::endl;
    std::cout << "row: ";
    std::cin >> row;
    std::cout << "column: ";
    std::cin >> column;
}