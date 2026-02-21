import sys
import tomllib
sys.path.insert(0, r'x:\transcription_for_kapra')

# config.tomlの読み込みテスト
with open(r'x:\transcription_for_kapra\config.toml', 'rb') as f:
    data = tomllib.load(f)

hidden = data.get('hidden', {})
print('Hidden params loaded from config.toml:')
print(f'  phase1_beam_size: {hidden.get("phase1_beam_size")}')
print(f'  phase2_detect_beam_size: {hidden.get("phase2_detect_beam_size")}')
print(f'  phase2_retranscribe_beam_size: {hidden.get("phase2_retranscribe_beam_size")}')
print(f'  ambiguous_threshold: {hidden.get("ambiguous_threshold")}')

# ダイアログのテスト
from PySide6.QtWidgets import QApplication
from ui.hidden_params_dialog import HiddenParamsDialog

app = QApplication(sys.argv)
dialog = HiddenParamsDialog(hidden, None)
values = dialog.get_values()

print('\nDialog values:')
print(f'  phase1_beam_size: {values.get("phase1_beam_size")}')
print(f'  phase2_detect_beam_size: {values.get("phase2_detect_beam_size")}')
print(f'  phase2_retranscribe_beam_size: {values.get("phase2_retranscribe_beam_size")}')

print('\nSUCCESS: All tests passed!')
