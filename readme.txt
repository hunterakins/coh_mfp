Hi user.
So you want to perform so range-coherent matched field processing on the SWellEx96 data set?
You've come to the right place. 

<<Step 1>>: is to pre-process the SWellEx96 set to get the snapshot representation.
This is carried out using swellex.audio.make_snapshots
Then run make_all_snapshot_for_rcmfp()
This forms snapshots for three sections, called sec1, sec2, and sec3, corresponding
to minutes 1.65 to 16.8, 23.5 to 37, and 40 to 52
sec3 is also considered the default section, so sometimes that tail is ommitted in processing

<<Step 2>>: Get velocity estimates.
To perform range-coherent mfp, you'll need to run velocity estimation for the
synthetic aperture length you're interested in
To do so, run vel_estimation.py with the appropriate switches.

For example, for the tonal set dragged from the deep source, an 8 element synth array,
for sec1, run 
python vel_estimation.py s5_deep 8 8 _sec1

This says used the N_fft = 16384 (8*2048 = 16384) snapshots
With an 8 synthetic element array
On the _sec1 segment for s5_deep toal set
This does incoherent sum stacked over frequency for all the tones in s5_deep 
The only choices for the factor are 1 and 8
are 1 and 8
You can add more by modifying make_all_snapshot_for_rcmfp() in swellex.audio

Generally, it's python vel_estimation.py [proj_str] [factor] [num_synth_els] [segment_tail]
AKA run vel_estimation on synthetic arrays of num_synth_els on snapshots formed with N_fft = factor*2048,
for tonals from proj_str, and on segment associated with segment_tail
Grab a cup of coffee, it might take a few minutes.


<<Step 3>>: Get tilt estimates.
Run tilt_estimation.py
Grab a cup of coffee, this will definitely take a while.
It will show you what the correlations for the different potential tilt angles are
You can use these correlations to specify tilt manually in the next step
By default, the tilt estimation does not use a synthetic array.
After running once, you can comment out "run_tilt_mfp(sec)" (line 46), and then
easily access the plotted results. 
My conclusion is that segment 3 should use tilt angle -1, whereas
segment 1 and 2 should use tilt angle -.5


<<Step 4>>: Run RCMFP
Now that you have velocity and tilt estimates you can do some RCMFP.
Use wnc_test.py to perform both incoherent Bartlett summing and white noise constraint
beamforming.

How? Follow this example:
Example
>> python wnc_test.py s5_deep 8 -.5 5 _sec2
Will produce ambiguity surfaces for a 5 element synthetic array for both
incoherent bartlett stacks for tonals in s5_deep with -.5 white noise gain constraint applied to 
segment 2 snapshots with 2048*8 = 16384 FFT length


<<Step 5>>:Analyze the results
comparison_plot.py will produce a panel plot for a specific covariance integration period
On the left will be the classic processing outputs and on the right will be the 
rcmfp outputs.
Top row is Bartlett, bottom row is WNC
All the parameters are specified in the top and include:
cov_index, synth_cov_index : which covariance matrix to examine
N_fft - base n_fft used in classic processing
fact - factor by which synthetic aperture has lengthened the permissible fft
seg_tail - which data segment is being examined
num_synth_els - how many synthetic elements
wn_gain, synth_wn_gain
There's no cmd line interface, so just modify the function and run



