#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <halconcpp/HalconCpp.h>
#include <halconcpp/HDevThread.h>

#include <QMainWindow>


using namespace HalconCpp;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

struct OCRResult
{
    QVector<QRect> ResultRect1;
    QVector<double> ResultPhi;
    QVector<QString> ResultStr;
    void clear()
    {
        ResultRect1.clear();
        ResultPhi.clear();
        ResultStr.clear();
    }
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
public:
    HTuple hv_DeepOcrHandle;
    QString m_ImgPath;
    OCRResult m_ResultRect2;


private slots:
    void on_PB_initModel_clicked();

    void on_PB_Infer_clicked();

    void on_PB_OpenIMG_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
