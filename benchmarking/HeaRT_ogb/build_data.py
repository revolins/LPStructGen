import subprocess

#split_type = ["CN_0_1_2", "CN_0_2_4", "CN_0_3_5", "SP_00_21_26", \
    #"SP_00_26_51", "SP_00_26_36", "PA_00_26_51", "PA_00_22_26", "PA_00_22_34" ]

def run_command(d, s, v, t):
    cmd_list = ["python", "gen_synth.py", "--data_name", d, \
            "--split_type", s, "--valid_rat", str(v), "--test_rat", str(t), "--debug_heur"]
    subprocess.run(cmd_list)
    cmd_list.append("--inverse")
    subprocess.run(cmd_list)

def main():
    data_name = ["ogbl-ppa", "ogbl-collab"] #"FF","SBM" "ogbl-collab", "ogbl-collab", 
    for d in data_name:
        
        # s = "CN"
        # v, t = 1, 2
        # run_command(d, s, v, t)
        # v, t = 2, 4
        # run_command(d, s, v, t)
        # v, t = 3, 5
        # run_command(d, s, v, t)
        
        # s = "PA"
        # v, t = 0.26, 0.34
        # run_command(d, s, v, t)
        # v, t = 0.22, 0.26
        # run_command(d, s, v, t)
        # v, t = 0.22, 0.34
        # run_command(d, s, v, t)
        
        s = "SP"
        v, t = 0.21, 0.30
        run_command(d, s, v, t)
        v, t = 0.20, 0.26
        run_command(d, s, v, t)
        v, t = 0.20, 0.33
        run_command(d, s, v, t)
        # v, t = 0.17, 0.26
        # run_command(d, s, v, t)
        # v, t = 0.26, 0.36
        # run_command(d, s, v, t)
                
    
if __name__ == "__main__":
    main()

