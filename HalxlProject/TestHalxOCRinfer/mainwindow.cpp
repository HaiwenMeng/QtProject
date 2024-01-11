#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QDebug"
#include  "QFileDialog"
#include "QElapsedTimer"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_PB_initModel_clicked()
{
    // Local control variables
    HTuple  hv_BestModelFilename, hv_BestModelDeepOCRFilename;
    HTuple  hv_Alphabet, hv_DLDeviceHandles;
    HTuple  hv_DLDevice;
    HTuple tDLDevice, tDeviceName, tDeviceType;
    hv_BestModelFilename = HTuple("F:/QtProject/BIN/")+"model_punched_numbers_best.pkl";
    hv_BestModelDeepOCRFilename = HTuple("F:/QtProject/BIN/")+"model_punched_numbers_best.pth";

    CreateDeepOcr(HTuple(), HTuple(), &hv_DeepOcrHandle);

    // 0-9.
    hv_Alphabet = HTuple::TupleGenSequence(0,9,1)+"";
    SetDeepOcrParam(hv_DeepOcrHandle, "recognition_alphabet", hv_Alphabet);

    //cpu
    QueryAvailableDlDevices("runtime", "cpu", &hv_DLDeviceHandles);
    GetDlDeviceParam(hv_DLDeviceHandles, "name", &tDeviceName);
    qDebug()<<QString(tDeviceName[0].S().ToUtf8());

    if (0 != (int((hv_DLDeviceHandles.TupleLength())==0)))
    {
        throw HException("No supported device found to continue this example.");
    }
    hv_DLDevice = ((const HTuple&)hv_DLDeviceHandles)[0];

    //SetDeepOcrParam(hv_DeepOcrHandle, "device", hv_DLDeviceHandles[0]);


    SetDeepOcrParam(hv_DeepOcrHandle, "detection_model", hv_BestModelFilename);
    SetDeepOcrParam(hv_DeepOcrHandle, "detection_batch_size", 1);
    SetDeepOcrParam(hv_DeepOcrHandle, "detection_optimize_for_inference", "true");
    qDebug()<<"000000000000000000000";


    try
    {
        GetDeepOcrParam(hv_DeepOcrHandle, "detection_device", &tDLDevice);
        qDebug()<<"222222222";
        if (0 != tDLDevice)
        {
            qDebug()<<"name:";
            GetDlDeviceParam(tDLDevice, "name", &tDeviceName);
            qDebug()<<QString(tDeviceName[0].S().ToUtf8());

            qDebug()<<"type:";

            GetDlDeviceParam(tDLDevice, "type", &tDeviceType);

            qDebug()<<QString(tDeviceType[0].S().ToUtf8());

        }
        ReadDeepOcr(hv_BestModelDeepOCRFilename, &hv_DeepOcrHandle);
        SetDeepOcrParam(hv_DeepOcrHandle, "device", hv_DLDevice);
        qDebug()<<"111111111";
    }

    catch(HException error)
    {
        qDebug()<<error.ErrorMessage().ToUtf8();
        return;
    }





//    try
//    {
//        //SetDlModelParam(hv_DeepOcrHandle,"device", hv_DLDevice);

//        GetDeepOcrParam(hv_DeepOcrHandle, "detection_device", &tDLDevice);
//        qDebug()<<"222222222";
//        if (0 != tDLDevice)
//        {
//            qDebug()<<"name:";
//            GetDlDeviceParam(tDLDevice, "name", &tDeviceName);
//            qDebug()<<QString(tDeviceName[0].S().ToUtf8());

//            qDebug()<<"type:";

//            GetDlDeviceParam(tDLDevice, "type", &tDeviceType);

//            qDebug()<<QString(tDeviceType[0].S().ToUtf8());

//        }
//    }

//    catch(HException error)
//    {
//        qDebug()<<error.ErrorMessage().ToUtf8();
//        return;
//    }



}

void MainWindow::on_PB_Infer_clicked()
{
    HTuple  hv_DeepOcrResult, hv_DetectionMode, hv_RecognitionMode;
    HTuple  hv_Words, hv_Width, hv_Height, hv_Wordsrow, hv_Wordscol;
    HTuple  hv_Wordsphi, hv_Wordslength1, hv_Wordslength2, hv_HasRecognition;
    HTuple  hv_RecognizedWord;



    HObject ho_Image;
    if(m_ImgPath.isEmpty())
    {
        qDebug()<< "img is empty";
        //return;
    }

    m_ResultRect2.clear();
    QElapsedTimer timer;
    timer.start();
    ReadImage(&ho_Image, "F:/QtProject/BIN/punched_numbers_00.jpg");

    //ReadImage(&ho_Image, HTuple(m_ImgPath.toStdString().c_str()));
    ApplyDeepOcr(ho_Image, hv_DeepOcrHandle, HTuple(), &hv_DeepOcrResult);

    GetDictParam(hv_DeepOcrResult, "key_exists", "words", &hv_DetectionMode);
    GetDictParam(hv_DeepOcrResult, "key_exists", "word", &hv_RecognitionMode);
    //
    if (0 != hv_DetectionMode)
    {
        //inputsize =  512*384*3

        GetDictTuple(hv_DeepOcrResult, "words", &hv_Words);
        //       GetImageSize(ho_Image, &hv_Width, &hv_Height);
        GetDictTuple(hv_Words, "row", &hv_Wordsrow);
        GetDictTuple(hv_Words, "col", &hv_Wordscol);
        GetDictTuple(hv_Words, "phi", &hv_Wordsphi);
        GetDictTuple(hv_Words, "length1", &hv_Wordslength1);
        GetDictTuple(hv_Words, "length2", &hv_Wordslength2);

        for(int i=0; i<hv_Wordsrow.Length(); i++)
        {
            QRect tRect = QRect(hv_Wordsrow[i].D(), hv_Wordscol[i].D(), hv_Wordslength1[i].D(),hv_Wordslength2[i].D());
            m_ResultRect2.ResultRect1.append(tRect);
            m_ResultRect2.ResultPhi.append(hv_Wordsphi[i].D());
        }

        GetDictParam(hv_Words, "key_exists", "word", &hv_HasRecognition);
        if (0 != hv_HasRecognition)
        {
            GetDictTuple(hv_Words, "word", &hv_RecognizedWord);
        }
        for(int i=0; i<hv_Wordsrow.Length(); i++)
        {
            m_ResultRect2.ResultStr.append(QString(hv_RecognizedWord[i].S().ToUtf8()));
        }
    }

    for(int i=0; i<m_ResultRect2.ResultStr.length(); i++)
    {
        qDebug()<<"OCR: "<<i<<m_ResultRect2.ResultStr[i];
    }
    qDebug()<<"run time"<<timer.elapsed()<<"msec";
}

void MainWindow::on_PB_OpenIMG_clicked()
{
    //    ui->label->setPixmap(QPixmap(QString("F:/QtProject/BIN/punched_numbers_00.jpg")));
    //return;

    m_ImgPath = QFileDialog::getOpenFileName();
    QImage *tImg = new QImage;
    QPixmap tpixmap;
    if(tImg->load(m_ImgPath))
    {
        ui->label->setPixmap(tpixmap.fromImage(*tImg));

    }
    else
    {
        qDebug()<<"load failed"<<m_ImgPath;
        QFile file(m_ImgPath);
        if(file.open(QIODevice::ReadOnly))
        {
            tpixmap.loadFromData(file.readAll());
            qDebug()<<"load over";
            ui->label->setPixmap(tpixmap);
        }
    }

}
