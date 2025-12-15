import pandas as pd

nvtx_csv_filepath = "nsysreps/test_02_PD_metrics/nvtx_gpu_proj_trace/trace.csv"
df_nvtx = pd.read_csv(nvtx_csv_filepath)
df_nvtx = df_nvtx[['Text','Start','End']]
df_nvtx[df_nvtx.Text.str.match('prompt_[0-9][0-9][0-9]') | df_nvtx.Text.str.match('token_[0-9][0-9][0-9]')]
df_nvtx['Duration'] = df_nvtx['End'] - df_nvtx['Start']
print(df_nvtx.columns)
print(df_nvtx.head())

cuda_mem_csv_filepath = "nsysreps/test_02_PD_metrics/_cuda_gpu_trace.csv"
df_mem = pd.read_csv(cuda_mem_csv_filepath)
df_mem = df_mem[['Start (ns)', 'Duration (ns)', 'Name']]
df_mem.columns = ['Start', 'Duration', 'Name']
df_mem['End'] = df_mem['Start'] + df_mem['Duration']
df_mem = df_mem[['Name', 'Start', 'End', 'Duration']]
#df_mem = df_mem[(df_mem.Name == '[CUDA memcpy Host-to-Device]') | (df_mem.Name == '[CUDA memcpy Device-to-Host]')]
#print(df_mem[(df_mem.Name == '[CUDA memcpy Device-to-Host]')])

