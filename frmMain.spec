# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['frmMain.py'],
             pathex=['F:\\vinAI\\water-level'],
             binaries=[],
             datas=[],
             hiddenimports=["pygubu","pygubu.builder.tkstdwidgets","pygubu.builder.widgets","pygubu.builder.widgets.scrolledframe"],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [('W ignore', None, 'OPTION')],
          exclude_binaries=True,
          name='WaterLevelDetection',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='frmMain')
