from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QPointF, QPoint, Qt
from PyQt5.QtGui import QWheelEvent
import os, sys, shutil
tdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, tdir)
sys.path.insert(0, os.path.dirname(tdir))
cdir = os.path.dirname(os.path.dirname(__file__))

def drag_bl1(qtbot, plot_widget):
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(433, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(270, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(270, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(470, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(470, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(470, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(650, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(650, 300))
    qtbot.wait(50)

def drag_bl2(qtbot, plot_widget):
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(439, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(270, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(270, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(475, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(475, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(480, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(650, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(650, 300))
    qtbot.wait(50)

def test_MDC_Fitter_1(qtbot, monkeypatch):
    from tool.MDC_Fitter import main
    # 模擬 QMessageBox.information 自動回傳 QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'), ''))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0_test_mfit.npz'), ''))
    
    file = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5')
    win = main(file=file)
    qtbot.waitExposed(win)
    win.load_file()
    qtbot.wait(100)
    qtbot.keyClick(win, QtCore.Qt.Key_Right)
    qtbot.keyClick(win, QtCore.Qt.Key_Left)
    qtbot.wait(100)
    
    
    win.maf1.setText('a')
    win.maf2.setText('a')
    win.mwf1.setText('a')
    win.mwf2.setText('a')
    win.mxf1.setText('a')
    win.mxf2.setText('a')
    win.maf1.setText('')
    win.maf2.setText('')
    win.mwf1.setText('')
    win.mwf2.setText('')
    win.mxf1.setText('')
    win.mxf2.setText('')
    win.maf1.setText('1')
    win.maf2.setText('1')
    win.mwf1.setText('1')
    win.mwf2.setText('1')
    win.mxf1.setText('-1')
    win.slider.setValue(537)
    win.fmreject()
    qtbot.wait(100)
    win.slider.setValue(544)
    win.fmreject()
    qtbot.wait(100)
    win.fmfall()
    qtbot.wait(5000)
    
    win.slider.setValue(200)
    win.fmrmv(test=True)
    qtbot.wait(100)
    win.fmrmv(test=True)
    qtbot.wait(100)
    win.fmcgl2()
    plot_widget = win.plot.viewport()
    center = plot_widget.rect().center()

    drag_bl1(qtbot, plot_widget)
    win.slider.setValue(400)
    win.fmcgl2()
    qtbot.wait(100)
    win.slider.setValue(520)
    qtbot.wait(100)
    win.fmrmv(test=True)
    qtbot.wait(100)
    win.fmrmv(test=True)
    qtbot.wait(100)
    drag_bl2(qtbot, plot_widget)
    qtbot.wait(100)
    win.slider.setValue(399)
    qtbot.wait(100)
    win.fmposcst()
    qtbot.wait(100)
    win.fmfall()
    qtbot.wait(10000)
    
    
    qtbot.keyClick(win, QtCore.Qt.Key_Down)
    qtbot.wait(500)
    qtbot.keyClick(win, QtCore.Qt.Key_Down)
    qtbot.wait(500)
    win.fmreject()
    qtbot.wait(100)
    win.fmreject()
    qtbot.wait(100)
    win.fmaccept()
    qtbot.wait(100)
    win.slider.setValue(544)
    qtbot.keyClick(win, QtCore.Qt.Key_Down)
    qtbot.wait(500)
    qtbot.keyClick(win, QtCore.Qt.Key_Down)
    qtbot.wait(500)
    
    qtbot.mouseClick(win.b_pos, Qt.LeftButton)
    qtbot.wait(100)
    qtbot.mouseClick(win.b_pos, Qt.LeftButton)
    qtbot.wait(100)
    
    
    win.slider.setValue(520)
    win.fmcgl2()
    win.fmcgl2()
    qtbot.wait(100)
    qtbot.mouseMove(plot_widget, pos=QPoint(450, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=center)
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(400, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(450, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=center)
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(450, 300))
    qtbot.keyClick(win, QtCore.Qt.Key_Up)
    qtbot.wait(500)
    qtbot.keyClick(win, QtCore.Qt.Key_Down)
    qtbot.wait(500)
    qtbot.keyClick(win, QtCore.Qt.Key_Enter)
    qtbot.wait(500)
    qtbot.keyClick(win, QtCore.Qt.Key_Z, Qt.ControlModifier)
    qtbot.wait(500)
    qtbot.keyClick(win, QtCore.Qt.Key_Y, Qt.ControlModifier)
    qtbot.wait(500)
    
    win.mfcomp1()
    qtbot.mouseMove(plot_widget, pos=QPoint(408, 205))
    qtbot.wait(50)
    win.mouse_clicked_event(event=None)
    
    qtbot.mouseMove(plot_widget, pos=QPoint(350, 170))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(380, 180))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(408, 300))
    qtbot.wait(50)
    
    qtbot.mouseMove(plot_widget, pos=QPoint(408, 210))
    qtbot.wait(50)
    win.mouse_clicked_event(event=None)
    win.mfcomp1()
    qtbot.wait(100)
    
    win.mfcomp2()
    qtbot.mouseMove(plot_widget, pos=QPoint(506, 205))
    qtbot.wait(50)
    win.mouse_clicked_event(event=None)
    
    qtbot.mouseMove(plot_widget, pos=QPoint(550, 170))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(470, 180))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(540, 300))
    qtbot.wait(50)
    
    qtbot.mouseMove(plot_widget, pos=QPoint(506, 210))
    qtbot.wait(50)
    win.mouse_clicked_event(event=None)
    win.mfcomp2()
    qtbot.wait(100)
    win.ffitcp()
    
    
    win.mfcomp1()
    qtbot.wait(100)
    qtbot.mouseMove(plot_widget, pos=QPoint(408, 205))
    qtbot.wait(50)
    win.mouse_clicked_event(event=None)
    
    qtbot.mouseMove(plot_widget, pos=QPoint(350, 170))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(380, 180))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(408, 300))
    qtbot.wait(50)
    win.mfbgd()
    
    win.mfcomp2()
    qtbot.wait(100)
    qtbot.mouseMove(plot_widget, pos=QPoint(506, 205))
    qtbot.wait(50)
    win.mouse_clicked_event(event=None)
    
    qtbot.mouseMove(plot_widget, pos=QPoint(550, 170))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(470, 180))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(540, 300))
    qtbot.wait(50)
    win.mfbgu()
    
    win.mfcomp1()
    qtbot.wait(100)
    win.mfcomp2()
    qtbot.wait(100)
    win.mfcomp1()
    qtbot.wait(100)
    win.mfcomp1()
    qtbot.wait(100)
    
    win.slider.setValue(win.index-1)
    qtbot.wait(100)
    win.slider.setValue(win.index-1)
    qtbot.wait(100)
    win.mflind()
    qtbot.wait(100)
    win.mflind()
    qtbot.wait(100)
    win.mflind()
    qtbot.wait(100)
    win.mflind()
    qtbot.wait(100)
    win.mfrind()
    qtbot.wait(100)
    win.mfrind()
    qtbot.wait(100)
    win.mfrind()
    qtbot.wait(100)
    win.mfrind()
    qtbot.wait(100)
    win.slider.setValue(win.index+1)
    qtbot.wait(100)
    win.slider.setValue(win.index+1)
    qtbot.wait(100)
    
    win.fmpreview()
    qtbot.waitExposed(win.tg)
    win.fmpreview()
    qtbot.waitExposed(win.tg)
    win.fmresidual()
    win.fmarea()
    win.fmfwhm()
    win.fmimse()
    win.fmresidual()
    win.fmarea()
    win.fmfwhm()
    win.fmimse()
    win.g_residual.close()
    win.g_area.close()
    win.g_fwhm.close()
    win.g_imse.close()
    
    
    win.fmend()
    qtbot.waitExposed(win.g_exp)
    win.fmend()
    qtbot.waitExposed(win.g_exp)
    win.fmend(1)
    qtbot.waitExposed(win.g_exp)
    win.fmend(2)
    qtbot.waitExposed(win.g_exp)
    win.savemfit()
    qtbot.wait(100)
    
    
    win.slider.setValue(200)
    qtbot.wait(100)
    win.fmrmv(test=True)
    qtbot.wait(100)
    win.slider.setValue(520)
    qtbot.wait(100)
    win.fmrmv(test=True)
    qtbot.wait(2)
    
    win.toggle_grid(checked=True)
    qtbot.wait(100)
    win.toggle_grid(checked=False)
    qtbot.wait(100)
    win.toggle_histogram()
    qtbot.wait(100)
    win.toggle_histogram()
    qtbot.wait(100)
    win.toggle_histogram()
    qtbot.wait(100)
    win.reset_histogram()
    qtbot.wait(100)
    win.auto_level_histogram()
    qtbot.wait(100)
    
    win.help_window()
    
    win.show_shortcuts()
    win.close()

def test_MDC_Fitter_2(qtbot, monkeypatch):
    from tool.MDC_Fitter import main
    # 模擬 QMessageBox.information 自動回傳 QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'), ''))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0_test_mfit.npz'), ''))
    file = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5')
    
    win = main(file=file)
    qtbot.waitExposed(win)
    
    win.slider.setValue(552)
    qtbot.wait(100)
    win.mfitplot()
    win.slider.setValue(0)
    qtbot.wait(100)
    win.fmrmv(test=True)
    qtbot.wait(100)
    win.slider.setValue(658)
    qtbot.wait(100)
    win.fmrmv(test=True)
    qtbot.wait(100)
    win.slider.setValue(545)
    qtbot.wait(100)
    
    plot_widget = win.plot.viewport()
    # 468-607
    qtbot.mouseMove(plot_widget, pos=QPoint(468, 300))
    qtbot.wait(50)
    qtbot.mousePress(plot_widget, Qt.LeftButton, pos=QPoint(468, 300))
    qtbot.wait(50)
    qtbot.mouseMove(plot_widget, pos=QPoint(607, 300))
    qtbot.wait(50)
    qtbot.mouseRelease(plot_widget, Qt.LeftButton, pos=QPoint(607, 300))
    qtbot.wait(50)
    
    win.slider.setValue(544)
    qtbot.wait(100)
    win.slider.setValue(545)
    qtbot.wait(100)
    
    win.close()
    
def test_MDC_Fitter_3(qtbot, monkeypatch):
    from tool.MDC_Fitter import main
    # 模擬 QMessageBox.information 自動回傳 QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'), ''))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0_test_mfit.npz'), ''))
    file = os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5')
    
    win = main(file=file, src='MDC_cut')
    qtbot.waitExposed(win)
    win.close()

def test_MDC_Fitter_4(qtbot, monkeypatch):
    from tool.MDC_Fitter import main
    # 模擬 QMessageBox.information 自動回傳 QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'), ''))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0_test_mfit.npz'), ''))
    
    file = os.path.join(os.path.dirname(__file__), 'data_cut.npz')
    win = main(file=file)
    qtbot.waitExposed(win)
    win.close()

def test_DataViewer(qtbot, monkeypatch):
    from tool.DataViewer import SliceBrowser, get_hwnd, disp_zarr_save, load_zarr
    # 模擬 QMessageBox.information 自動回傳 QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (os.path.join(os.path.dirname(__file__), 'test_save.zarr'), ''))
    
    hwnd = get_hwnd()
    assert isinstance(hwnd, int)
    path = os.path.join(os.path.dirname(__file__), 'test_cube.zarr')
    output = os.path.join(os.path.dirname(__file__), 'test_cube_disp.zarr')
    mode, shape, xmin, xmax, ymin, ymax, E = load_zarr(path)
    disp_zarr_save(path, output, shape, max_val=10750)
    
    path = os.path.join(os.path.dirname(__file__), 'test_cube.zarr')
    win = SliceBrowser(path=path, hwnd=hwnd)
    qtbot.waitExposed(win)
    win.on_radio_button_changed("E")
    qtbot.wait(100)
    win.on_radio_button_changed("kx")
    qtbot.wait(100)
    win.export_slice()
    qtbot.wait(100)
    win.on_radio_button_changed("ky")
    qtbot.wait(100)
    win.export_slice()
    qtbot.wait(100)
    win.rotate_slider.setValue(90)
    win.sync_rotate_edit()
    qtbot.wait(100)
    win.apply_rotation()
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'test_save.zarr'), ignore_errors=True)
    win.export_slice()
    win.on_radio_button_changed("kx")
    qtbot.wait(100)
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'test_save.zarr'), ignore_errors=True)
    win.export_slice()
    shutil.rmtree(os.path.join(os.path.dirname(__file__), 'test_save.zarr'), ignore_errors=True)
    win.save_as_zarr_disp()
    win.close()
    
    win = SliceBrowser(os.path.join(os.path.dirname(__file__), 'test_save.zarr'), hwnd)
    qtbot.waitExposed(win)
    win.bin_e_spin.setValue(5)
    win.on_bin_change()
    win.on_radio_button_changed("kx")
    win.on_bin_change()
    win.on_radio_button_changed("ky")
    win.on_bin_change()
    win.close()
    
    shutil.rmtree(os.path.join(path, '__disp__.zarr'), ignore_errors=True)
    win = SliceBrowser(path=path, hwnd=hwnd)
    qtbot.waitExposed(win)
    win.close()

def test_RawDataViewer(qtbot, monkeypatch):
    from tool.RawDataViewer import main
    from tool.loader import loadfiles
    # 模擬 QMessageBox.information 自動回傳 QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QDialog, 'exec_', lambda self: QMessageBox.Ok)
    path = []
    path.append(os.path.join(os.path.dirname(__file__), 'simulated_R1_15.0_R2_0#id#0d758f03.h5'))
    path.append(os.path.join(os.path.dirname(__file__), 'UPSPE20_2_test_1559#id#3cf2122d.json'))
    lfs = loadfiles(path, name='internal')
    win = main(lfs, test=True)
    qtbot.waitExposed(win)
    event = QtCore.QEvent(QtCore.QEvent.MouseButtonPress)
    win.energy_mode(event)
    qtbot.wait(100)
    
    # 模擬鼠標移動到 plot widget
    plot_widget = win.plot.viewport()
    center = plot_widget.rect().center()
    plot_widgetx = win.plotx.viewport()
    centerx = plot_widgetx.rect().center()
    
    qtbot.mouseMove(plot_widget, pos=QPoint(250, 230))
    qtbot.wait(100)
    
    # 模擬鼠標移動到 plot widget
    qtbot.mouseMove(plot_widget, pos=center)
    qtbot.wait(100)
    
    # 模擬鼠標點擊
    qtbot.mouseClick(plot_widget, Qt.LeftButton, pos=center)
    qtbot.wait(100)
    
    # 模擬鼠標移動
    qtbot.mouseMove(plot_widgetx, pos=centerx)
    qtbot.wait(100)
    
    win.range_changed()
    qtbot.wait(100)
    
    
    # 模擬鼠標移動到 plot widget
    qtbot.mouseMove(plot_widget, pos=QPoint(250, 230))
    qtbot.wait(100)
    
    # 模擬鼠標點擊
    qtbot.mouseClick(plot_widget, Qt.LeftButton, pos=QPoint(250, 230))
    qtbot.wait(100)
    
    # 模擬鼠標移動到 plot widget
    qtbot.mouseMove(plot_widget, pos=QPoint(255, 235))
    qtbot.wait(100)
    
    # 模擬鼠標移動
    qtbot.mouseMove(plot_widgetx, pos=centerx)
    qtbot.wait(100)
    
    win.range_changed()
    qtbot.wait(100)
    
    wheel_event = QWheelEvent(
        QPointF(center),  # pos (滑鼠位置)
        QPointF(center),  # globalPos (全局位置)
        QPoint(0, 0),     # pixelDelta
        QPoint(0, 120),   # angleDelta (正值向上滾動,負值向下滾動)
        Qt.NoButton,      # buttons
        Qt.NoModifier,    # modifiers
        Qt.ScrollUpdate,  # phase
        False            # inverted
    )
    win.text_display.wheelEvent(wheel_event)
    qtbot.wait(100)
    
    wheel_event = QWheelEvent(
        QPointF(center),  # pos (滑鼠位置)
        QPointF(center),  # globalPos (全局位置)
        QPoint(0, 0),     # pixelDelta
        QPoint(0, -120),   # angleDelta (正值向上滾動,負值向下滾動)
        Qt.NoButton,      # buttons
        Qt.NoModifier,    # modifiers
        Qt.ScrollUpdate,  # phase
        False            # inverted
    )
    win.text_display.wheelEvent(wheel_event)
    qtbot.wait(100)

    win.load_file(path)
    qtbot.wait(100)
    win.close()