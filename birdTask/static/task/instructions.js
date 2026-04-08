// instructions.js

var instructions = [];
const style1 = "font-size:20px";

const INSTR_IMG = {
  intro_task: "../static/img/task_assets/reward/drone0.png",
  intro_controls: "../static/img/task_assets/shared/bucket1.png",
  memory_example: "../static/img/task_assets/practice/apple.jpg",
  reward_example: "../static/img/task_assets/reward/supply-bag.png",
  loss_example: "../static/img/task_assets/loss/hazard-bag.png"
};

var inst1_incorrect = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"><br><br> Some of your answers were incorrect. Some instructions will be repeated. Please pay attention!</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var inst_summary = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"> You did not answer all questions correctly. Some of the instructions will be repeated before you see the questions again. <br><br><br>Pay close attention!</p>',

    '<p style="font-size:20px"><img src="' + INSTR_IMG.intro_task + '" width="40%"></img> <br><br><br> In this game, drones drop supply bags onto the ground. Your goal is to position your collector to catch as many supplies as you can.</p>',

    '<p style="font-size:20px"><img src="' + INSTR_IMG.intro_controls + '" width="40%"></img> <br><br><br> You should use right and left arrow keys to move the collector.</p>',

    '<p style="font-size:20px"> After you position the collector, the sky will darken slightly. At this time you can no longer move the collector.</p>',

    '<p style="font-size:20px"> You will then see the drone dropping a supply bag.</p>',

    '<p style="font-size:20px"> The bag breaks open near the ground and the supplies scatter.</p>',

    '<p style="font-size:20px"> Your score is determined by the number of items you catch in your collector.<br><br><br>If you align your collector perfectly, you will catch all ten items! Otherwise, you will catch fewer depending on how far off your collector is.</p>',

    '<p style="font-size:20px"> A new turn begins when the screen lights up again. At this time you are once again able to move the collector.</p>',

    '<p style="font-size:20px"> If you do not move the collector on one or two turns, we assume you are happy with its position. <br><br> However, you should not leave the collector in one place for more than a few turns. If you do, we will warn you, and if you persist, we may have to end the game early!</p>',

    '<p style="font-size:20px"> The bag will fall near the drone, but the exact position will vary around the drone because of unpredictable winds!</p>',

    '<p style="font-size:20px"> You may have noticed that the drone can also move unpredictably. The best prediction for its position on one turn is its position on the previous turn, but it may fly to a new location at any time.</p>',

    '<p style="font-size:20px"><strong> Your best strategy is to position the collector directly under where you think the drone is located.</strong></p>',

    '<p style="font-size:20px"> Remember though: in the real game, you cannot actually see the drone, only the supply bag that it drops!</p>',

    '<p style="font-size:20px"> Your movement of the collector is exactly the same as before — but you have to estimate where the drone is located based on where it has been.</p>',

    '<p style="font-size:20px"><img src="' + INSTR_IMG.memory_example + '" width="40%"></img> <br><br><br>You will also notice that in each turn, a distinct item will appear where the supplies land. <br><br>There will be a memory test based on those items.</p>',

    '<p style="font-size:20px"> The full game will have 4 different environments. Each environment will also have a different drone with different flying behavior.</p>',

    '<p style="font-size:20px"> You will now see the questions about the game again. Please answer all questions correctly, or we will have to end the game.</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var inst3_incorrect = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"><br><br> You did not respond. <br><br> We must terminate the game here.</p>',
  ],
  show_clickable_nav: false,
};

var inst1 = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"><img src="' + INSTR_IMG.intro_task + '" width="40%"></img> <br><br><br> In this game, drones drop supply bags onto the ground. Your goal is to catch as many supplies as you can by moving your collector to where you think the supplies will land.</p>',

    '<p style="font-size:20px"><img src="' + INSTR_IMG.intro_controls + '" width="40%"></img> <br><br><br> You should use right and left arrow keys to move the collector.</p>',

    '<p style="font-size:20px"><br><br> Now give it a try. Make a response by using the left or right arrow key.</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var inst2 = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"> After you position the collector, the sky will darken slightly. At this time you can no longer move the collector.</p>',
    '<p style="font-size:20px"> You will then see the drone dropping a supply bag.</p>',
    '<p style="font-size:20px"> The bag breaks open near the ground and the supplies scatter.</p>',
    '<p style="font-size:20px"> Your score is determined by the number of items that you catch in your collector.<br><br><br> If you align your collector perfectly, you will catch all ten items! Otherwise, you will catch fewer depending on how far off your collector is.</p>',
    '<p style="font-size:20px"> A new turn begins when the screen lights up again. At this time you are once again able to move the collector.</p>',
    '<p style="font-size:20px"> Remember: when the screen is clear, you can move the collector. When it is slightly darkened, the collector is frozen.</p>',
    '<p style="font-size:20px"> If you do not move the collector on one or two turns, we assume you are happy with its position. <br><br> However, you should not leave the collector in one place for more than a few turns. If you do, we will warn you, and if you persist, we may have to end the game early!</p>',
    '<p style="font-size:20px"> Now give it a try. Notice that you can only move your collector when the screen is clear. <br><br><br>You can see each turn how many items you caught by the number from 0 to 10 that appears on the screen.</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var inst3 = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"> The bag will fall near the drone, but the exact position will vary around the drone because of unpredictable winds!</p>',
    '<p style="font-size:20px"> The bag might fall in front of the drone,</p>',
    '<p style="font-size:20px"> or it might fall just under the drone,</p>',
    '<p style="font-size:20px"> or it might fall behind the drone.</p>',
    '<p style="font-size:20px">You may have noticed that the drone can also move unpredictably. The best prediction for its position on one turn is its position on the previous turn, but it may fly to a new location at any time.</p>',
    '<p style="font-size:20px"><strong>Your best strategy is to position the collector directly under where you think the drone is located.</strong></p>',
    '<p style="font-size:20px"> Now that you know how the game works, try playing a few turns, noticing that the drone moves and where the bag falls.</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var inst4 = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"> We are almost ready for the full game, but there are just a few important differences. <br>Most importantly, in the real game, you cannot actually see the drone, only the supply bag that it drops!</p>',
    '<p style="font-size:20px">Your movement of the collector is exactly the same as before — but you have to estimate where the drone is located based on where it has been.</p>',
    '<p style="font-size:20px"><img src="' + INSTR_IMG.memory_example + '" width="40%"></img> <br><br><br>You will also notice that in each turn, a distinct item will appear where the supplies land.</p>',
    '<p style="font-size:20px"> Now, try a few turns where the drone is not visible, also noting the item that appears on each turn.</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var inst5 = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"> The full game will have 4 different environments with 4 different wind conditions. Each of the 4 environments will also have a different drone with different flying behavior.</p>',

    '<p style="font-size:20px"><img src="' + INSTR_IMG.reward_example + '" width="40%"></img> <br><br><br><strong>In some environments, accurate placement helps you gain more points.</strong> The items you catch are valuable supplies, and each one adds to your score.</p>',

    '<p style="font-size:20px"><img src="' + INSTR_IMG.loss_example + '" width="40%"></img> <br><br><br><strong>In other environments, accurate placement helps you lose fewer points.</strong> The items are hazardous, and each one you fail to catch with your shield costs you points.</p>',

    '<p style="font-size:20px"> You will be reminded each time the environment changes.</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var inst6 = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"> Following each of the 4 environments, you will complete a memory task based on the items that appeared. <br>You will be asked which item in a pair appeared first, how many items you believe appeared between them, and for some pairs, where a middle item occurred on a timeline between the two.</p>',
    '<p style="font-size:20px"> For the timeline question, you will see a slider with the two boundary items at each end and the middle item above. Move the slider to indicate where in time the middle item appeared relative to the two boundary items.</p>',
    '<p style="font-size:20px"> You should note these items as they appear but you do not need to memorize them. <br>Remember: your primary goal is to maximize your score by catching as many items in your collector as you can!</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var inst7 = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"> The memory task will follow each environment.</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var ready = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"> We are now beginning the game.<br><br>Good luck!</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var quiz = {
  type: 'instructions',
  pages: [
    '<p style="font-size:20px"> You will now see some questions about the game. You should answer all of them correctly to proceed.<br><br>Good luck!</p>',
  ],
  show_clickable_nav: true,
  button_label_previous: "Prev",
  button_label_next: "Next"
};

var num_loops = 0;
var comprehension1 = {
  type: 'comprehension1',
};

var comprehension2 = {
  type: 'comprehension2',
};