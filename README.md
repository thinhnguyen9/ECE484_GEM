# Autonomous lane keeping

A perception-planning-control pipeline tested on a real-life GEM e2 car. Based on the ECE 484 starter code that implements a baseline lane detection and Pure Pursuit controller.

We proposed a new model with steering and throttle dynamics, and a full-state feedback control law using LQR and state estimation using Kalman filter. This improves cross-track RMS error by 18% and velocity RMS error by 61% compared with the given Pure Pursuit controller.

The small clip shows the car following a curved track and braking autonomously when encountering a traffic cone:
<p align="center">
  <img src="./media/demo_inside.gif" alt="Estimation error demo" width="480" />
</p>

## Do note:
For information on e.g., what rostopics to record, software versions, hardware specs, etc. PLEASE, please look at the vehicle documentation, linked on the [course project page](https://publish.illinois.edu/safe-autonomy/resources/) and [directly in the CfA resources here](https://publish.illinois.edu/robotics-autonomy-resources/).

When recording rosbags, you can change the buffer size if you get warnings about packets dropping; [read the cli](https://wiki.ros.org/rosbag/Commandline#record).

The majority of the code that you will be working with in this starting code can be found at the path `e#/src/vehicle_drivers/gem_gnss_control/scripts`. The README within that specific directory is tailored for explaining the usage of the `gem_gnss_tracker_pid.py`+ `lane_detection.py` system. 

If you want to use the baseline lane following, you will need to download the corresponding model weights as a `.pt` file. You can read the README.md at the directory mentioned in the line above for instruction on this. 

The ROS environment for e2 vs e4 is very similar. There are just a few small differences, e.g., 4 arena (corner) cameras on e4 versus a front-left front-right only corner cameras on e2. The actual python code (not various CMake configs), is nearly identical, aside from potential some parameters about vehicle length/width, etc.

You are not constrained to using a PID controller (as is done in `gem_gnss_tracker_pid.py` for lane following). We have also implemented Pure Pursuit and Stanley controllers in `gem_gnss_tracker_pp.py` and `gem_gnss_tracker_stanley_rtk.py` respectively. However, note that the Pure Pursuit and Stanley controllers do not currently support integration with the lane detection module / lane following. You would have to do some integration work (though, that should not be too hard).

You can choose to only use pieces of this starter code (or, you can choose to use none of it). Purely up to you what you want to do

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.engr.illinois.edu/GolfCar/gem-workspace-and-starter-code.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.engr.illinois.edu/GolfCar/gem-workspace-and-starter-code/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
