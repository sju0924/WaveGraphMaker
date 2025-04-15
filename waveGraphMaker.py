import colorama
from colorama import Fore, Style
import pandas as pd
import cmd
import sys
from Step1 import Step1 
from Step2 import Step2_acc, Step2_SF, Step2_srss
from Step3 import Step3_acc,  Step3_srss, Step3_eq
from Step4 import Step4_acc, Step4_SF, Step4_srss

class SubMenuForStep1(cmd.Cmd):
    
    intro = (
        Fore.CYAN
        + "Step 1\n"
        + "1. 목표 응답 스펙트럼 그래프 그리기\n"
        + "(back) 뒤로 가기\n"
        + Style.RESET_ALL
    )
    prompt = Fore.GREEN + "Step1> " + Style.RESET_ALL

    def do_1(self, arg):
        """서브 메뉴의 1번 기능"""
        Step1()
        print(Fore.MAGENTA + ">>> output: /Step1" + Style.RESET_ALL)

    def do_back(self, arg):
        """
        메인 메뉴로 돌아가기
        'back' 명령어를 입력하면 서브 메뉴(cmdloop)를 빠져나옵니다.
        """
        print(Fore.YELLOW + ">>> 메인 메뉴로 돌아갑니다." + Style.RESET_ALL)
        return True  # True를 리턴하면 cmdloop()를 종료합니다.

    def default(self, line):
        """정의되지 않은 명령어 처리"""
        print(Fore.RED + f"알 수 없는 명령어입니다: {line}" + Style.RESET_ALL)

class SubMenuForStep2(cmd.Cmd):
    SF = []
    srss_table = None
    intro = (
        Fore.CYAN
        + "Step 2\n"
        + "(1) Scale up Factor 구하기\n"
        + "(2) 응답 스펙트럼 그래프 그리기\n"
        + "(3) 시간-가속도 그래프 그리기\n"
        + "(back) 뒤로 가기\n"
        + Style.RESET_ALL
    )
    prompt = Fore.GREEN + "Step2 > " + Style.RESET_ALL

    def do_1(self, arg):
        cmd = 'N'
        while True:
            cmd = input(Fore.MAGENTA + ">>> input: waves.xlsx의 data Sheet에 응답 스펙트럼 데이터를 입력하였습니까? (y/N)" + Style.RESET_ALL)
            if cmd == 'y' or cmd =='Y':
                break
        self.srss_table, self.SF = Step2_SF()
        print(Fore.MAGENTA + ">>> 현재 Scale up Factor: " + Style.RESET_ALL, self.SF, "\n")


    def do_2(self, arg):
        if len(self.SF) == 0:
            print(Fore.MAGENTA + "응답 스펙트럼 그래프를 그리기 위해서는 1번 메뉴에서 Scale up Factor를 불러와야 합니다.\n" + Style.RESET_ALL)
            return False
        
        Step2_srss(self.srss_table, self.SF)
        print(Fore.MAGENTA + ">>> output: Step2/SRSS, Step2/SRSS_scale" + Style.RESET_ALL)

    def do_3(self, arg):
        if len(self.SF) == 0:
            print(Fore.MAGENTA + "시간-가속도 그래프를 그리기 위해서는 1번 메뉴에서 Scale up Factor를 불러와야 합니다.\n" + Style.RESET_ALL)
            return False
        
        Step2_acc(self.SF)
        print(Fore.MAGENTA + ">>> output: Step2/Acceleration" + Style.RESET_ALL)

    def do_back(self, arg):
        """
        메인 메뉴로 돌아가기
        'back' 명령어를 입력하면 서브 메뉴(cmdloop)를 빠져나옵니다.
        """
        print(Fore.YELLOW + ">>> 메인 메뉴로 돌아갑니다." + Style.RESET_ALL)
        return True  # True를 리턴하면 cmdloop()를 종료합니다.

    def default(self, line):
        """정의되지 않은 명령어 처리"""
        print(Fore.RED + f"알 수 없는 명령어입니다: {line}" + Style.RESET_ALL)

class SubMenuForStep3(cmd.Cmd):

    prompt = Fore.GREEN + "Step3 > " + Style.RESET_ALL
    intro = (
    Fore.CYAN
    + "Step 3\n"
    + "(1) 측정한 지진파에 대한 .eq 파일 생성하기\n"
    + "(2) 응답 스펙트럼 그래프 그리기\n"
    + "(3) 시간-가속도 그래프 그리기\n"
    + "(back) 뒤로 가기\n"
    + Style.RESET_ALL
)

    def do_1(self, arg):
        cmd = 'N'
        while True:
            cmd = input(Fore.MAGENTA + ">>> input: waves.xlsx의 data Sheet에 응답 스펙트럼 데이터를 입력하였습니까? (y/N)" + Style.RESET_ALL)
            if cmd == 'y' or cmd =='Y':
                break
        Step3_eq()
        print(Fore.MAGENTA + ">>> output: Step3/output/eq" + Style.RESET_ALL, "\n")


    def do_2(self, arg):
        while True:
            cmd = input(Fore.MAGENTA + ">>> input: Step3/input/SGS_result에 응답 스펙트럼 데이터를 입력하였습니까? (y/N)" + Style.RESET_ALL)
            if cmd == 'y' or cmd =='Y':
                break
        Step3_srss()
        print(Fore.MAGENTA + ">>> output: Step3/output/SRSS, Step3/output/SRSS_scale" + Style.RESET_ALL)

    def do_3(self, arg):
        while True:
            cmd = input(Fore.MAGENTA + ">>> input: Step3/input/Shake_M_result에 시간-가속도 스펙트럼 데이터를 입력하였습니까? (y/N)" + Style.RESET_ALL)
            if cmd == 'y' or cmd =='Y':
                break        
        Step3_acc()
        print(Fore.MAGENTA + ">>> output: Step3/output/Acceleration" + Style.RESET_ALL)

    def do_back(self, arg):
        """
        메인 메뉴로 돌아가기
        'back' 명령어를 입력하면 서브 메뉴(cmdloop)를 빠져나옵니다.
        """
        print(Fore.YELLOW + ">>> 메인 메뉴로 돌아갑니다." + Style.RESET_ALL)
        return True  # True를 리턴하면 cmdloop()를 종료합니다.

    def default(self, line):
        """정의되지 않은 명령어 처리"""
        print(Fore.RED + f"알 수 없는 명령어입니다: {line}" + Style.RESET_ALL)

class SubMenuForStep4(cmd.Cmd):
    
    SF = []
    srss_table = None
    intro = (
        Fore.CYAN
        + "Step 4\n"
        + "(1) Scale up Factor 구하기\n"
        + "(2) 응답 스펙트럼 그래프 그리기\n"
        + "(3) 시간-가속도 그래프 그리기\n"
        + "(back) 뒤로 가기\n"
        + Style.RESET_ALL
    )
    prompt = Fore.GREEN + "Step4 > " + Style.RESET_ALL

    def do_1(self, arg):
        cmd = 'N'
        while True:
            cmd = input(Fore.MAGENTA + ">>> input: Step4/input/SGS_result에 응답 스펙트럼 데이터를 입력하였습니까? (y/N)" + Style.RESET_ALL)
            if cmd == 'y' or cmd =='Y':
                break
        self.srss_table, self.SF = Step4_SF()
        print(Fore.MAGENTA + ">>> 현재 Scale up Factor: " + Style.RESET_ALL, self.SF, "\n")


    def do_2(self, arg):
        if len(self.SF) == 0:
            print(Fore.MAGENTA + "응답 스펙트럼 그래프를 그리기 위해서는 1번 메뉴에서 Scale up Factor를 불러와야 합니다.\n" + Style.RESET_ALL)
            return False
        
        Step4_srss(self.srss_table, self.SF)
        print(Fore.MAGENTA + ">>> output: Step4/output/SRSS, Step4/output/SRSS_scale" + Style.RESET_ALL)

    def do_3(self, arg):
        if len(self.SF) == 0:
            print(Fore.MAGENTA + "시간-가속도 그래프를 그리기 위해서는 1번 메뉴에서 Scale up Factor를 불러와야 합니다.\n" + Style.RESET_ALL)
            return False
        while True:
            cmd = input(Fore.MAGENTA + ">>> input: Step4/input/RSP_match_result에 시간-가속도 스펙트럼 데이터를 입력하였습니까? (y/N)" + Style.RESET_ALL)
            if cmd == 'y' or cmd =='Y':
                break
        Step4_acc(self.SF)
        print(Fore.MAGENTA + ">>> output: Step4/output/Acceleration" + Style.RESET_ALL)

    def do_back(self, arg):
        """
        메인 메뉴로 돌아가기
        'back' 명령어를 입력하면 서브 메뉴(cmdloop)를 빠져나옵니다.
        """
        print(Fore.YELLOW + ">>> 메인 메뉴로 돌아갑니다." + Style.RESET_ALL)
        return True  # True를 리턴하면 cmdloop()를 종료합니다.

    def default(self, line):
        """정의되지 않은 명령어 처리"""
        print(Fore.RED + f"알 수 없는 명령어입니다: {line}" + Style.RESET_ALL)
class MyCLI(cmd.Cmd):
    intro = (
        Fore.CYAN
        + r"""
 _ _ _   _   _ _  ___    __  ___   _   ___  _ _   _   _   _   _  _ ___  ___ 
| | | | / \ | | || __|  / _|| o \ / \ | o \| U | | \_/ | / \ | |//| __|| o \
| V V || o || V || _|  ( |_n|   /| o ||  _/|   | | \_/ || o ||  ( | _| |   /
 \_n_/ |_n_| \_/ |___|  \__/|_|\\|_n_||_|  |_n_| |_| |_||_n_||_|\\|___||_|\\
                                                                            
                 W A V E   G R A P H   M A K E R
"""
        + Style.RESET_ALL
        + "\n" 
        + Fore.YELLOW
        + "수행할 단계를 입력하세요.\n"
        + "(1) Step1\n"
        + "(2) Step2\n"
        + "(3) Step3\n"
        + "(4) Step4\n"
        + Style.RESET_ALL
    )
    
    # 명령 프롬프트에 표시될 문자열
    prompt = Fore.GREEN + "Command > " + Style.RESET_ALL
    
    def do_1(self, arg):
        """1번 명령어를 처리하는 메소드"""
        self.step1()
    
    def do_2(self, arg):
        """2번 명령어를 처리하는 메소드"""
        self.step2()
    def do_3(self, arg):
        """2번 명령어를 처리하는 메소드"""
        self.step3()
    def do_4(self, arg):
        """2번 명령어를 처리하는 메소드"""
        self.step4()
    
    def step1(self):
        print(Fore.MAGENTA + ">>> Step 1" + Style.RESET_ALL)
        sub_menu = SubMenuForStep1()
        sub_menu.cmdloop()  
        
    def step2(self):
        sub_menu = SubMenuForStep2()
        sub_menu.cmdloop()  

        
    def step3(self):
        sub_menu = SubMenuForStep3()
        sub_menu.cmdloop()  

    def step4(self):
        sub_menu = SubMenuForStep4()
        sub_menu.cmdloop() 

        # 실제 로직은 여기서 작성
    
    def do_exit(self, arg):
        """프로그램을 종료합니다."""
        print(Fore.RED + "프로그램을 종료합니다. 안녕히 가세요!" + Style.RESET_ALL)
        return True  # True를 리턴하면 cmdloop를 빠져나와 종료
    
    def default(self, line):
        """정의되지 않은 명령어가 들어왔을 때 처리"""
        print(Fore.RED + f"알 수 없는 명령어입니다: {line}" + Style.RESET_ALL)

if __name__ == '__main__':
    try:
        colorama.init()
        MyCLI().cmdloop()
    except KeyboardInterrupt:
        print("\n" + Fore.RED + "강제 종료하였습니다." + Style.RESET_ALL)
        sys.exit(0)