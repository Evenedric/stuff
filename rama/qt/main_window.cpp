
#include <QDesktopServices>
#include <QUrl>
#include <QFileDialog>
#include <QFile>
#include <QByteArray>
#include <QTimer>
#include <QClipboard>

#include "main_window.h"
#include "ui_main_window.h"
#include "about.h"
#include "sweep.h"
#include "../version.h"
#include "../../toolkit/error.h"

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent), ui(new Ui::MainWindow), autorun_(true),
  watcher_(this)
{
  ui->setupUi(this);

  // Set some menu and control defaults.
  ui->actionAntialiasing->setChecked(true);

  // Configure docked windows.
  setTabPosition(Qt::AllDockWidgetAreas, QTabWidget::North);
  tabifyDockWidget(ui->dock_model, ui->dock_plot);
  tabifyDockWidget(ui->dock_model, ui->dock_antenna);
  tabifyDockWidget(ui->dock_model, ui->dock_script_messages);
  resizeDocks({ui->dock_model}, {650}, Qt::Horizontal);   //@@@ scale width by main window size
  ui->dock_model->raise();

  // Default shows and hidden controls.
  ui->display_style_Exy->hide();
  ui->display_style_TE->hide();
  ui->display_style_TM->hide();
  ui->mode_label->hide();
  ui->mode_number->hide();

  // Connect the model viewer to other controls.
  ui->model->Connect(ui->script_messages, ui->parameter_pane, ui->plot,
                     ui->dock_model, ui->dock_plot, ui->dock_script_messages);
  ui->model->Connect2(ui->antenna_plot);

  // Set up the file watcher.
  QObject::connect(&watcher_, &QFileSystemWatcher::fileChanged,
                   this, &MainWindow::OnFileChanged);
}

MainWindow::~MainWindow() {
  delete ui;
}

void MainWindow::LoadFile(const QString &full_path) {
  {
    auto list = watcher_.files();
    if (!list.empty()) {
      watcher_.removePaths(list);   // Generates a warning if list empty
    }
  }
  script_filename_ = full_path;
  if (ReloadScript(true)) {
    // Watch this file for changes.
    if (!watcher_.addPath(script_filename_)) {
      Error("Can not watch %s for changes", script_filename_.toUtf8().data());
    }
  }
}

bool MainWindow::ReloadScript(bool rerun_even_if_same) {
  if (script_filename_.isEmpty()) {
    Error("No script has been opened yet");
    return false;
  }

  // Open and read the file.
  QFile f(script_filename_);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    Error("Can not open %s", script_filename_.toUtf8().data());
    return false;
  }
  QByteArray buffer = f.readAll();
  if (buffer.isEmpty()) {
    // Either the file was empty or an error occurred.
    // @@@ Distinguish between these cases.
    // @@@ If the file is empty, it's possible we loaded it just as the
    //     editor was saving it.
    Error("Can not read %s", script_filename_.toUtf8().data());
    return false;
  }

  // Run the script.
  bool script_ran = ui->model->RunScript(buffer.data(), rerun_even_if_same);

  // Highlight the "current model" label, revert it to normal color after 1s.
  if (script_ran) {
    QFileInfo fi(script_filename_);
    ui->current_model->setText(fi.baseName());
    ui->current_model->setStyleSheet("QLabel { background-color : yellow; }");
    QTimer::singleShot(1000, this, &MainWindow::ResetCurrentModelBackground);
  }

  // Show the correct controls based on the cavity type.
  ui->display_style_Ez->setVisible(ui->model->IsEzCavity());
  ui->display_style_Exy->setVisible(ui->model->IsExyCavity());
  ui->display_style_TE->setVisible(ui->model->IsTEMode());
  ui->display_style_TM->setVisible(ui->model->IsTMMode());
  ui->animate->setVisible(ui->model->NumWaveguideModes() == 0);
  ui->mode_number->setVisible(ui->model->NumWaveguideModes());
  ui->mode_label->setVisible(ui->model->NumWaveguideModes());
  ui->mode_number->setMinimum(0);
  ui->mode_number->setMaximum(std::max(0, ui->model->NumWaveguideModes() - 1));
  ui->model->SetWaveguideModeDisplayed(ui->mode_number->value());

  return true;
}

void MainWindow::OnFileChanged(const QString &path) {
  // We might get this event more than once for a file modification (it depends
  // on how the editor saves the file). To try and avoid reading the file in
  // some in-between state, we trigger the actual file read 100ms from now.
  QTimer::singleShot(100, this, &MainWindow::ReloadTimeout);
}

void MainWindow::ReloadTimeout() {
  if (autorun_) {
    ReloadScript(false);
  }
}

void MainWindow::ResetCurrentModelBackground() {
  ui->current_model->setStyleSheet("QLabel { }");
}

void MainWindow::on_actionQuit_triggered() {
  QApplication::quit();
}

void MainWindow::on_actionAbout_triggered() {
  auto about = new About(this);
  about->show();
}

void MainWindow::on_actionSweep_triggered() {
  Sweep sweep(this);
  sweep.SetupForModel(ui->model);
  if (sweep.exec() == QDialog::Accepted) {
    ui->model->Sweep(sweep.GetParameterName().toUtf8().data(),
                     sweep.GetStartValue(),
                     sweep.GetEndValue(),
                     sweep.GetNumSteps());
  }
}

void MainWindow::on_actionOpen_triggered() {
  QString filename = QFileDialog::getOpenFileName(this, "Open model",
      QString(), "Lua files (*.lua)", Q_NULLPTR, QFileDialog::ReadOnly);
  if (!filename.isEmpty()) {
    LoadFile(filename);
  }
}

void MainWindow::on_actionReload_triggered() {
  ReloadScript(true);
}

void MainWindow::on_actionAutoRun_triggered() {
  autorun_ = !autorun_;
  if (autorun_) {
      ReloadScript(false);
  }
}

void MainWindow::on_actionExportBoundaryDXF_triggered() {
  QString filename = QFileDialog::getSaveFileName(this,
      "Select a DXF file to save", QString(), "DXF files (*.dxf)");
  if (!filename.isEmpty()) {
    ui->model->ExportBoundaryDXF(filename.toUtf8().data());
  }
}

void MainWindow::on_actionExportBoundaryXY_triggered() {
  QString filename = QFileDialog::getSaveFileName(this,
      "Select a file to save", QString(), "*.*");
  if (!filename.isEmpty()) {
    ui->model->ExportBoundaryXY(filename.toUtf8().data());
  }
}

void MainWindow::on_actionExportFieldAsMatlab_triggered() {
  QString filename = QFileDialog::getSaveFileName(this,
      "Select a matlab file to save", QString(), "*.mat");
  if (!filename.isEmpty()) {
    ui->model->ExportFieldMatlab(filename.toUtf8().data());
  }
}

void MainWindow::on_actionExportPlotAsMatlab_triggered() {
  QString filename = QFileDialog::getSaveFileName(this,
      "Select a matlab file to save", QString(), "*.mat");
  if (!filename.isEmpty()) {
    ui->model->ExportPlotMatlab(filename.toUtf8().data());
  }
}

void MainWindow::on_actionCopyParameters_triggered() {
  ui->model->CopyParametersToClipboard();
}

void MainWindow::on_actionZoomExtents_triggered() {
  ui->model->ZoomExtents();
}

void MainWindow::on_actionZoomIn_triggered() {
  ui->model->Zoom(1.0/sqrt(2));
}

void MainWindow::on_actionZoomOut_triggered() {
  ui->model->Zoom(sqrt(2));
}

void MainWindow::on_actionViewLinesAndPorts_triggered() {
  ui->model->ToggleShowBoundary();
}

void MainWindow::on_actionViewVertices_triggered() {
  ui->model->ToggleShowBoundaryVertices();
}

void MainWindow::on_actionViewVertexDerivatives_triggered() {
  ui->model->ToggleShowBoundaryDerivatives();
}

void MainWindow::on_actionShowGrid_triggered() {
  ui->model->ToggleGrid();
}

void MainWindow::on_actionShowMarkers_triggered() {
  ui->model->ToggleShowMarkers();
}

void MainWindow::on_actionAntialiasing_triggered() {
  ui->model->ToggleAntialiasing();
}

void MainWindow::on_actionEmitTrace_triggered() {
  ui->model->ToggleEmitTraceReport();
}

void MainWindow::on_actionSelectLevenbergMarquardt_triggered() {
  ui->model->SetOptimizer(LEVENBERG_MARQUARDT);
}

void MainWindow::on_actionSelectSubspaceDogleg_triggered() {
  ui->model->SetOptimizer(SUBSPACE_DOGLEG);
}

void MainWindow::on_actionStartOptimization_triggered() {
  ui->model->Optimize();
}

void MainWindow::on_actionStopSweepOrOptimization_triggered() {
  ui->model->StopSweepOrOptimize();
}

void MainWindow::on_actionRamaManual_triggered() {
  /*@@@ link to manual that is in the resource bundle or installed location
  wxString path = wxStandardPaths::Get().GetResourcesDir();
  wxURI url("file://" + path + "/rama.html");
  wxLaunchDefaultBrowser(url.BuildURI(), wxBROWSER_NEW_WINDOW);
  */
}

void MainWindow::on_actionRamaWebsite_triggered() {
  QString link = __APP_URL__;
  QDesktopServices::openUrl(QUrl(link));
}

void MainWindow::on_actionLuaManual_triggered() {
  QString link = "http://www.lua.org/manual/5.3/";
  QDesktopServices::openUrl(QUrl(link));
}

void MainWindow::on_actionCheckForLatestVersion_triggered() {
  /*@@@ deal with this:
  wxMessageDialog dlg (this, "Clicking 'Yes' will quit the application and start "
      "downloading the installer for the latest version of " __APP_NAME__
      ". Do you want to proceed?", "Install latest version?",
      wxYES_NO | wxNO_DEFAULT | wxCENTRE);
  if (dlg.ShowModal() == wxID_YES) {
      #if __APPLE__
      wxLaunchDefaultBrowser(__APP_URL__ __APP_LATEST_MAC_DOWNLOAD_PATH__, wxBROWSER_NEW_WINDOW);
      #else
      wxLaunchDefaultBrowser(__APP_URL__ __APP_LATEST_WINDOWS_DOWNLOAD_PATH__, wxBROWSER_NEW_WINDOW);
      #endif
      exit(0);
  }
  */
}

void MainWindow::on_reload_resets_parameters_stateChanged(int arg1) {
  ui->model->SetIfRunScriptResetsParameters(arg1);
}

void MainWindow::on_show_field_stateChanged(int arg1) {
  // Turning off the field view automatically turns off the animation.
  if (!arg1) {
    if (ui->animate->isChecked()) {
      ui->animate->setChecked(false);
      ui->model->Animate(0);
    }
  }
  ui->model->ViewField(arg1);
}

void MainWindow::on_animate_stateChanged(int arg1) {
  // Turning on the animation automatically turns on the field view.
  if (arg1) {
    if (!ui->show_field->isChecked()) {
      ui->show_field->setChecked(true);
      ui->model->ViewField(1);
    }
  }
  ui->model->Animate(arg1);
}

void MainWindow::on_color_map_currentIndexChanged(int index) {
  ui->model->SetDisplayColorScheme(index);
}

void MainWindow::on_brightness_valueChanged(int value) {
  ui->model->SetBrightness(value);
}

void MainWindow::on_plot_type_currentIndexChanged(int index) {
  ui->model->SelectPlot(index);
}

void MainWindow::on_mesh_type_currentIndexChanged(int index) {
  ui->model->ViewMesh(index);
}

void MainWindow::on_antenna_show_stateChanged(int arg1) {
  ui->model->ToggleAntennaShow();
}

void MainWindow::on_antenna_scale_max_stateChanged(int arg1) {
  ui->model->ToggleAntennaScaleMax();
}

void MainWindow::on_mode_number_valueChanged(int arg1) {
  ui->model->SetWaveguideModeDisplayed(arg1);
}

void MainWindow::on_display_style_Ez_currentIndexChanged(int index) {
  UpdateDisplayStyle(index);
}

void MainWindow::on_display_style_Exy_currentIndexChanged(int index) {
  UpdateDisplayStyle(index);
}

void MainWindow::on_display_style_TM_currentIndexChanged(int index) {
  UpdateDisplayStyle(index);
}

void MainWindow::on_display_style_TE_currentIndexChanged(int index) {
  UpdateDisplayStyle(index);
}

void MainWindow::UpdateDisplayStyle(int index) {
  // All display style choice boxes are synchronized with each other and with
  // the display style state in the model.
  // @@@ Do we need to sync the TM and TE display styles too? they only have 5 entries.
  // @@@ Does TM/TE display style sync when a mode model is loaded?
  ui->display_style_Exy->setCurrentIndex(index);
  ui->display_style_Ez->setCurrentIndex(index);
  ui->model->SetDisplayStyle(index);
}

void MainWindow::on_copy_to_clipboard_clicked() {
  // Copy script messages to clipboard.
  auto *list = ui->script_messages;
  int n = list->count();
  QString s;
  for (int i = 0; i < n; i++) {
    s += list->item(i)->text() + "\n";
  }
  QGuiApplication::clipboard()->setText(s);
}