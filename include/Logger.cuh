#include <stdio.h>
enum logPriority{critical, warn, error, info, none};

void prc(logPriority l)
{
	switch (l)
	{
	case critical:
		printf("\033[1;31m"); //Set the text to the color red.
		break;
	case warn:
		break;
	case error:
		break;
	case info:
		break;
	default:
		printf("\033[0m"); //Resets the text to default color.
		break;
	}
}



#define Log(l_, f_, ...) prc(l_); printf((f_), __VA_ARGS__); prc(logPriority::none)