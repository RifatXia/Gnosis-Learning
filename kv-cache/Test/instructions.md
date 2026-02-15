we would be working on @phase3b_experiment.py and you need make these corrections to the code            
- first adapt everything from @phase2_experiment.py                                                       
- then the flow needs to be this way which would get repeated 10 times:                                                       
- load a,b (warmup) -> ask a,b,q (cache hit) -> clear cache                                                                   
- load a,b (warmup -> ask b, q (cache miss) -> clear cache   
- make all those changes to the loops, and the main experiment so that it reflects the flow well                                                                 
                                                                                                                                
ensure and fully ensure that the KV cache is cleared so that there is no contamination, and the plots would be like           
before but the json for the warmup should have the warmup for the 20 times as per the repeat and hence the plot would be      
having a std deviation of the 20 warmups, the ttft and throughput remains the same