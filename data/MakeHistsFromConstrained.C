#include "TString.h"
#include "TFile.h"
#include "TH1.h"
#include "TGraphAsymmErrors.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TF1.h"
#include <TMatrixD.h>
#include "TLine.h"
#include "TStyle.h"
#include <iostream>
#include <fstream>

TH1D* rescaleaxis(TH1D* h, double scale, int new_bins=0, double offset=0.0){
    //This function rescales the x-axis.
    //TH1D* h_new = (TH1D*)h->Clone();
    //h_new->Reset();
    TH1D* h_new;
    if (new_bins){
      h_new = new TH1D(h->GetName(), h->GetTitle(), new_bins, h->GetXaxis()->GetXmin()*scale + offset, (new_bins*scale) + offset);
    }else{
      h_new = new TH1D(h->GetName(), h->GetTitle(), h->GetNbinsX(), (h->GetXaxis()->GetXmin()*scale) + offset, (h->GetXaxis()->GetXmax()*scale) + offset);
    }
    int N = h->GetNbinsX();
    for (int i = 0; i < N; i++){
        double scaled_x = (double)h->GetBinCenter(i+1) * scale + offset;
        double content = h->GetBinContent(i+1);
        double error = h->GetBinError(i+1);
        h_new->Fill(scaled_x, content);
        h_new->SetBinError(i+1, error);
    }
    //h_new->GetXaxis()->SetBinLabel(N, "overflow");
    //h_new->RebinX();
    h_new->SetFillColor(h->GetFillColor());
    h_new->SetLineColor(h->GetLineColor());
    h_new->SetLineWidth(h->GetLineWidth());
    h_new->SetMarkerColor(h->GetMarkerColor());
    h_new->SetMarkerStyle(h->GetMarkerStyle());
    h_new->SetMarkerSize(h->GetMarkerSize());
    h_new->SetFillStyle(h->GetFillStyle());
    h_new->GetXaxis()->SetTitle(h->GetXaxis()->GetTitle());
    h_new->GetYaxis()->SetTitle(h->GetYaxis()->GetTitle());
    h_new->GetXaxis()->SetTitleOffset(h->GetXaxis()->GetTitleOffset());
    h_new->GetYaxis()->SetTitleOffset(h->GetYaxis()->GetTitleOffset());
    h_new->GetXaxis()->SetTitleSize(h->GetXaxis()->GetTitleSize());
    h_new->GetYaxis()->SetTitleSize(h->GetYaxis()->GetTitleSize());
    h_new->GetXaxis()->SetLabelSize(h->GetXaxis()->GetLabelSize());
    h_new->GetYaxis()->SetLabelSize(h->GetYaxis()->GetLabelSize());
    h_new->GetYaxis()->SetRangeUser(-0.02*h->GetBinContent(h->GetMaximumBin()), h->GetMaximum()*1.1);

    h_new->GetYaxis()->SetLabelFont(62);//22); //132);
    h_new->GetYaxis()->SetTitleFont(62);//22); //132);
    
    return h_new;
}

void rescaleaxisgraph(TGraphAsymmErrors* h, double scale){
    //This function rescales the x-axis.
    //TGraphAsymmErrors* h_new = (TGraphAsymmErrors*)h->Clone();
   // h_new->Clear();
    //new TGraphAsymmErrors(h->GetName(), h->GetTitle(), h->GetN(), h->GetXaxis()->GetXmin()*scale, h->GetXaxis()->GetXmax()*scale);
    int N = h->GetN();
    double *x = h->GetX();
    //auto y = h->GetY();
    for (int i = 0; i < N; i++){
         //h_new->SetPoint(i, x[i] * scale, y[i]);
          x[i]=x[i]*scale;
    }
    h->GetHistogram()->Delete();
    h->SetHistogram(0);
    //h_new->GetHistogram()->Rebin(100);
    //return h_new;
}

void MakeHistsFromConstrained(){

  TString file_name = "Xp_1500_data";
  float bin_start = 0.0;
  float bin_end = 1600.0;
  int num_bins = 16;
  
  TString variable = "Reconstructed Shower Energy";
  TString units = "MeV";//"MeV";
  //Shower Energy";//Shower Cos$\theta$";
  bool LEE = false;
  double chi_no = 7.06;
  double axis_scale = 100.0; //100.0; //0.22222222;
  double axis_offset = 150.0; //0.0; //-1.0;

  int new_bins = 0; //0; //9;

  gStyle->SetOptStat(0);
  gStyle->SetLegendBorderSize(0);
  gStyle->SetLegendFillColor(0);
  gStyle->SetLegendFont(62);//22); //132);
  gStyle->SetLegendTextSize(0.06);
  gStyle->SetLabelFont(62);//22); //132);
  gStyle->SetTitleFont(62);//22); //132);
  gStyle->SetLabelSize(0.06);
  gStyle->SetTitleSize(0.06);


  int color_no = kRed;
  int color_main = kBlue;
  int color_data = kBlack;

  TFile *fout = new TFile(file_name + "_hists.root", "RECREATE");
  TH1D *h_data = new TH1D("data", "Data Histogram", num_bins, bin_start, bin_end);
  TH1D *h_bkg = new TH1D("bkg", "Background Histogram", num_bins, bin_start, bin_end);
  TH1D *h_sig = new TH1D("sig", "Signal Histogram", num_bins, bin_start, bin_end);

  TFile *fin_main = new TFile(file_name + ".root");

  TMatrixD* pred_ptr = (TMatrixD*)fin_main->Get("matrix_Y_under_X");
  TMatrixD* data_ptr = (TMatrixD*)fin_main->Get("matrix_data_Y");
  TMatrixD* cov_ptr = (TMatrixD*)fin_main->Get("matrix_YY_under_XX");

  fin_main->Close();

  TMatrixD pred = *pred_ptr;
  TMatrixD data = *data_ptr;
  TMatrixD cov = *cov_ptr;

  pred.T();
  data.T();
  cov.T();

  int bins = pred.GetNcols();
  for (int i = 0; i < bins; i++){
    h_data->SetBinContent(i+1, data(0,i));
    h_bkg->SetBinContent(i+1, pred(0,i));
    h_sig->SetBinContent(i+1, pred(0,i));
  }

  fout->Write();
  fout->Close();

}
