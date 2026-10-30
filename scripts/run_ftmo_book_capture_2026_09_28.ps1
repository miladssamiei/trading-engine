$python = 'C:\Users\Milad\miniconda3\envs\reinforce_rl\python.exe'
$script = 'C:\Users\Milad\Desktop\RR\scripts\ftmo_book_source_capture.py'
$terminal = 'C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe'
$output = 'C:\Users\Milad\Desktop\RR\data\results\ftmo_book_source_qualification\2026-09-28_1520'
& $python $script --terminal $terminal --expected-server FTMO-Demo --symbols US500.cash XAUUSD --seconds 5400 --poll-seconds 1 --output $output
$captureExit = $LASTEXITCODE
& $python 'C:\Users\Milad\Desktop\RR\scripts\ftmo_book_source_analyze.py' --input $output
if ($captureExit -ne 0) { exit $captureExit }
exit $LASTEXITCODE
