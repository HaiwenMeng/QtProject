#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <halconcpp/HalconCpp.h>

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


public:




private slots:
    void on_pushButton_init_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
